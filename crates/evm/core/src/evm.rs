use std::ops::{Deref, DerefMut};

use crate::{
    backend::DatabaseExt, constants::DEFAULT_CREATE2_DEPLOYER_CODEHASH, Env, InspectorExt,
};
use alloy_consensus::constants::KECCAK_EMPTY;
use alloy_evm::{
    eth::EthEvmContext,
    precompiles::{DynPrecompile, PrecompileInput, PrecompilesMap},
    Evm, EvmEnv,
};
use alloy_primitives::{Address, Bytes, U256};
use foundry_fork_db::DatabaseError;
use revm::{
    context::{
        result::{EVMError, ExecResultAndState, ExecutionResult, HaltReason},
        BlockEnv, CfgEnv, ContextTr, CreateScheme, Evm as RevmEvm, JournalTr, LocalContext, TxEnv,
    },
    handler::{
        instructions::EthInstructions, EthFrame, EthPrecompiles, FrameResult,
        Handler, MainnetHandler,
    },
    inspector::InspectorHandler,
    interpreter::{
        interpreter::EthInterpreter, interpreter_action::FrameInit, return_ok, CallInput, CallInputs, CallOutcome, CallScheme,
        CallValue, CreateInputs, CreateOutcome, FrameInput, Gas, InstructionResult,
        InterpreterResult, SharedMemory,
    },
    precompile::{
        secp256r1::{P256VERIFY, P256VERIFY_BASE_GAS_FEE},
        PrecompileSpecId, Precompiles,
    },
    primitives::hardfork::SpecId,
    Context, Journal,
};

pub fn new_evm_with_inspector<'i, 'db, I: InspectorExt + ?Sized>(
    db: &'db mut dyn DatabaseExt,
    env: Env,
    inspector: &'i mut I,
) -> FoundryEvm<'db, &'i mut I> {
    let ctx = EthEvmContext {
        journaled_state: {
            let mut journal = Journal::new(db);
            journal.set_spec_id(env.evm_env.cfg_env.spec);
            journal
        },
        block: env.evm_env.block_env,
        cfg: env.evm_env.cfg_env,
        tx: env.tx,
        chain: (),
        local: LocalContext::default(),
        error: Ok(()),
    };
    let spec = ctx.cfg.spec;

    let mut evm = FoundryEvm {
        inner: RevmEvm::new_with_inspector(
            ctx,
            inspector,
            EthInstructions::default(),
            get_precompiles(spec),
        ),
    };

    inject_precompiles(&mut evm);

    evm
}

pub fn new_evm_with_existing_context<'a>(
    ctx: EthEvmContext<&'a mut dyn DatabaseExt>,
    inspector: &'a mut dyn InspectorExt,
) -> FoundryEvm<'a, &'a mut dyn InspectorExt> {
    let spec = ctx.cfg.spec;

    let mut evm = FoundryEvm {
        inner: RevmEvm::new_with_inspector(
            ctx,
            inspector,
            EthInstructions::default(),
            get_precompiles(spec),
        ),
    };

    inject_precompiles(&mut evm);

    evm
}

/// Conditionally inject additional precompiles into the EVM context.
fn inject_precompiles(evm: &mut FoundryEvm<'_, impl InspectorExt>) {
    if evm.inspector().is_odyssey() {
        evm.precompiles_mut().apply_precompile(P256VERIFY.address(), |_| {
            // Create a wrapper function that adapts the new API
            let precompile_fn = |input: PrecompileInput<'_>| -> Result<_, _> {
                P256VERIFY.precompile()(input.data, P256VERIFY_BASE_GAS_FEE)
            };
            Some(DynPrecompile::from(precompile_fn))
        });
    }
}

/// Get the precompiles for the given spec.
fn get_precompiles(spec: SpecId) -> PrecompilesMap {
    PrecompilesMap::from_static(
        EthPrecompiles {
            precompiles: Precompiles::new(PrecompileSpecId::from_spec_id(spec)),
            spec,
        }
        .precompiles,
    )
}

/// Get the call inputs for the CREATE2 factory.
fn get_create2_factory_call_inputs(
    salt: U256,
    inputs: &CreateInputs,
    deployer: Address,
) -> CallInputs {
    let calldata = [&salt.to_be_bytes::<32>()[..], &inputs.init_code[..]].concat();
    CallInputs {
        caller: inputs.caller,
        bytecode_address: deployer,
        target_address: deployer,
        scheme: CallScheme::Call,
        value: CallValue::Transfer(inputs.value),
        input: CallInput::Bytes(calldata.into()),
        gas_limit: inputs.gas_limit,
        is_static: false,
        return_memory_offset: 0..0,
    }
}

pub struct FoundryEvm<'db, I: InspectorExt> {
    #[allow(clippy::type_complexity)]
    pub inner: RevmEvm<
        EthEvmContext<&'db mut dyn DatabaseExt>,
        I,
        EthInstructions<EthInterpreter, EthEvmContext<&'db mut dyn DatabaseExt>>,
        PrecompilesMap,
        EthFrame<EthInterpreter>,
    >,
}

impl<I: InspectorExt> FoundryEvm<'_, I> {
    pub fn run_execution(
        &mut self,
        frame: FrameInput,
    ) -> Result<FrameResult, EVMError<DatabaseError>> {
        let mut handler = FoundryHandler::<_>::default();

        // Convert FrameInput to the appropriate frame init type
        match frame {
            FrameInput::Call(call_inputs) => {
                // Create a FrameInit with the call inputs
                let frame_init = FrameInit {
                    depth: 0, // Start at depth 0 for top-level execution
                    memory: SharedMemory::new(),
                    frame_input: FrameInput::Call(call_inputs),
                };
                let result = handler.inner.inspect_run_exec_loop(&mut self.inner, frame_init)?;
                Ok(result)
            }
            FrameInput::Create(create_inputs) => {
                // Handle CREATE2 factory override logic
                if let CreateScheme::Create2 { salt } = create_inputs.scheme {
                    if self.inner.inspector.should_use_create2_factory(&mut self.inner.ctx, &create_inputs) {
                        let gas_limit = create_inputs.gas_limit;
                        let create2_deployer = self.inner.inspector.create2_deployer();
                        let call_inputs = get_create2_factory_call_inputs(salt, &create_inputs, create2_deployer);
                        
                        // Sanity check that CREATE2 deployer exists
                        let code_hash = self.inner.journal_mut().load_account(create2_deployer)?.info.code_hash;
                        if code_hash == KECCAK_EMPTY {
                            return Ok(FrameResult::Call(CallOutcome {
                                result: InterpreterResult {
                                    result: InstructionResult::Revert,
                                    output: Bytes::copy_from_slice(
                                        format!("missing CREATE2 deployer: {create2_deployer}").as_bytes(),
                                    ),
                                    gas: Gas::new(gas_limit),
                                },
                                memory_offset: 0..0,
                            }));
                        } else if code_hash != DEFAULT_CREATE2_DEPLOYER_CODEHASH {
                            return Ok(FrameResult::Call(CallOutcome {
                                result: InterpreterResult {
                                    result: InstructionResult::Revert,
                                    output: "invalid CREATE2 factory output".into(),
                                    gas: Gas::new(gas_limit),
                                },
                                memory_offset: 0..0,
                            }));
                        }
                        
                        // Execute the call frame instead - create FrameInit with CallInputs
                        let call_frame_init = FrameInit {
                            depth: 0,
                            memory: SharedMemory::new(),
                            frame_input: FrameInput::Call(Box::new(call_inputs)),
                        };
                        let mut result = handler.inner.inspect_run_exec_loop(&mut self.inner, call_frame_init)?;
                        
                        // Convert call result back to create result
                        if let FrameResult::Call(mut call_result) = result {
                            let address = match call_result.instruction_result() {
                                return_ok!() => Address::try_from(call_result.output().as_ref())
                                    .map_err(|_| {
                                        call_result.result = InterpreterResult {
                                            result: InstructionResult::Revert,
                                            output: "invalid CREATE2 factory output".into(),
                                            gas: Gas::new(gas_limit),
                                        };
                                    })
                                    .ok(),
                                _ => None,
                            };
                            
                            result = FrameResult::Create(CreateOutcome { 
                                result: call_result.result, 
                                address 
                            });
                        }
                        
                        return Ok(result);
                    }
                }
                
                // Standard create execution - create FrameInit with CreateInputs
                let frame_init = FrameInit {
                    depth: 0,
                    memory: SharedMemory::new(),
                    frame_input: FrameInput::Create(create_inputs),
                };
                let result = handler.inner.inspect_run_exec_loop(&mut self.inner, frame_init)?;
                Ok(result)
            }
            FrameInput::Empty => {
                // Handle empty frame input - should not happen in normal execution
                Err(EVMError::Custom("Empty frame input not supported".into()))
            }
        }
    }
}

impl<'db, I: InspectorExt> Evm for FoundryEvm<'db, I> {
    type Precompiles = PrecompilesMap;
    type Inspector = I;
    type DB = &'db mut dyn DatabaseExt;
    type Error = EVMError<DatabaseError>;
    type HaltReason = HaltReason;
    type Spec = SpecId;
    type Tx = TxEnv;

    fn chain_id(&self) -> u64 {
        self.inner.ctx.cfg.chain_id
    }

    fn block(&self) -> &BlockEnv {
        &self.inner.block
    }

    fn db_mut(&mut self) -> &mut Self::DB {
        &mut self.inner.ctx.journaled_state.database
    }

    fn precompiles(&self) -> &Self::Precompiles {
        &self.inner.precompiles
    }

    fn precompiles_mut(&mut self) -> &mut Self::Precompiles {
        &mut self.inner.precompiles
    }

    fn inspector(&self) -> &Self::Inspector {
        &self.inner.inspector
    }

    fn inspector_mut(&mut self) -> &mut Self::Inspector {
        &mut self.inner.inspector
    }

    fn set_inspector_enabled(&mut self, _enabled: bool) {
        unimplemented!("FoundryEvm is always inspecting")
    }

    fn transact_raw(
        &mut self,
        tx: Self::Tx,
    ) -> Result<ExecResultAndState<ExecutionResult>, Self::Error> {
        self.inner.ctx.tx = tx;

        let mut handler = FoundryHandler::<_>::default();
        let result = handler.inspect_run(&mut self.inner)?;
        Ok(ExecResultAndState {
            result,
            state: self.inner.ctx.journaled_state.finalize(),
        })
    }

    fn transact_system_call(
        &mut self,
        _caller: Address,
        _contract: Address,
        _data: Bytes,
    ) -> Result<ExecResultAndState<ExecutionResult>, Self::Error> {
        unimplemented!()
    }

    fn finish(self) -> (Self::DB, EvmEnv<Self::Spec>)
    where
        Self: Sized,
    {
        let Context { block: block_env, cfg: cfg_env, journaled_state, .. } = self.inner.ctx;

        (journaled_state.database, EvmEnv { block_env, cfg_env })
    }
}

impl<'db, I: InspectorExt> Deref for FoundryEvm<'db, I> {
    type Target = Context<BlockEnv, TxEnv, CfgEnv, &'db mut dyn DatabaseExt>;

    fn deref(&self) -> &Self::Target {
        &self.inner.ctx
    }
}

impl<I: InspectorExt> DerefMut for FoundryEvm<'_, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.ctx
    }
}

pub struct FoundryHandler<'db, I: InspectorExt> {
    #[allow(clippy::type_complexity)]
    inner: MainnetHandler<
        RevmEvm<
            EthEvmContext<&'db mut dyn DatabaseExt>,
            I,
            EthInstructions<EthInterpreter, EthEvmContext<&'db mut dyn DatabaseExt>>,
            PrecompilesMap,
            EthFrame<EthInterpreter>,
        >,
        EVMError<DatabaseError>,
        EthFrame<EthInterpreter>,
    >,
    create2_overrides: Vec<(usize, CallInputs)>,
}

impl<I: InspectorExt> Default for FoundryHandler<'_, I> {
    fn default() -> Self {
        Self { inner: MainnetHandler::default(), create2_overrides: Vec::new() }
    }
}

impl<'db, I: InspectorExt> Handler for FoundryHandler<'db, I> {
    type Evm = RevmEvm<
        EthEvmContext<&'db mut dyn DatabaseExt>,
        I,
        EthInstructions<EthInterpreter, EthEvmContext<&'db mut dyn DatabaseExt>>,
        PrecompilesMap,
        EthFrame<EthInterpreter>,
    >;
    type Error = EVMError<DatabaseError>;
    type HaltReason = HaltReason;
}

impl<I: InspectorExt> InspectorHandler for FoundryHandler<'_, I> {
    type IT = EthInterpreter;
}
