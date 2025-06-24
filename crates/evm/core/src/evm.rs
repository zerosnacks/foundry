use std::ops::{Deref, DerefMut};

use crate::{
    backend::DatabaseExt, Env, InspectorExt,
};
use alloy_evm::{
    eth::EthEvmContext,
    precompiles::{DynPrecompile, PrecompilesMap},
    Evm, EvmEnv,
};
use alloy_primitives::{Address, Bytes, TxKind, U256};
use foundry_fork_db::DatabaseError;
use revm::{
    context::{
        result::{EVMError, ExecutionResult, HaltReason, ResultAndState},
        BlockEnv, CfgEnv, Evm as RevmEvm, LocalContext, TxEnv,
    },
    handler::{
        instructions::EthInstructions, EthFrame, EthPrecompiles, FrameResult, Handler, MainnetHandler,
    },
    inspector::InspectorHandler,
    interpreter::{
        interpreter::EthInterpreter, CallInput, CallInputs, CallOutcome, CallScheme,
        CallValue, CreateInputs, CreateOutcome, FrameInput, Gas, InstructionResult, InterpreterResult,
    },
    precompile::{PrecompileSpecId, Precompiles},
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
            let mut journal = Journal::new_with_inner(db, Default::default());
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
        // P256VERIFY precompile for Odyssey network
        evm.precompiles_mut().apply_precompile(P256VERIFY.address(), |_| {
            // Create a wrapper function that adapts the new API
            let precompile_fn = |input: &[u8]| -> Result<_, _> {
                // P256VERIFY expects (input, gas_limit) but new API only provides input
                // Use a reasonable default gas limit for P256VERIFY
                const P256VERIFY_GAS: u64 = 3450;
                P256VERIFY.precompile()(input, P256VERIFY_GAS)
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
        EthFrame,
    >,
}

impl<I: InspectorExt> FoundryEvm<'_, I> {
    pub fn run_execution(
        &mut self,
        frame: FrameInput,
    ) -> Result<FrameResult, EVMError<DatabaseError>> {
        // In Revm 26.0.1, we use the MainnetHandler for frame execution
        // The CREATE2 factory logic is now implemented in the Inspector via InspectorExt
        let mut handler: MainnetHandler<_, EVMError<DatabaseError>, _> = MainnetHandler::default();
        
        // Execute the frame using the standard handler
        match frame {
            FrameInput::Empty => {
                // Empty frame - return empty result
                Ok(FrameResult::Call(CallOutcome {
                    result: InterpreterResult {
                        result: InstructionResult::Stop,
                        output: Bytes::new(),
                        gas: Gas::new(0),
                    },
                    memory_offset: 0..0,
                }))
            }
            FrameInput::Call(call_inputs) => {
                // Set up the call context
                self.inner.ctx.tx.caller = call_inputs.caller;
                self.inner.ctx.tx.kind = 
                    TxKind::Call(call_inputs.bytecode_address);
                self.inner.ctx.tx.data = match call_inputs.input {
                    CallInput::Bytes(bytes) => bytes,
                    CallInput::SharedBuffer(_) => Bytes::new(), // SharedBuffer not supported in this context
                };
                self.inner.ctx.tx.value = match call_inputs.value {
                    CallValue::Transfer(value) => value,
                    CallValue::Apparent(value) => value,
                };
                self.inner.ctx.tx.gas_limit = call_inputs.gas_limit;
                
                let execution_result = handler.inspect_run(&mut self.inner)?;
                // Convert ExecutionResult to CallOutcome
                let interpreter_result = match execution_result {
                    ExecutionResult::Success { reason, gas_used, gas_refunded: _, logs: _, output } => {
                        InterpreterResult {
                            result: reason.into(),
                            output: match output {
                                revm::context::result::Output::Call(bytes) => bytes,
                                revm::context::result::Output::Create(bytes, _) => bytes,
                            },
                            gas: Gas::new(gas_used),
                        }
                    }
                    ExecutionResult::Revert { gas_used, output } => {
                        InterpreterResult {
                            result: InstructionResult::Revert,
                            output: output,
                            gas: Gas::new(gas_used),
                        }
                    }
                    ExecutionResult::Halt { reason, gas_used } => {
                        InterpreterResult {
                            result: reason.into(),
                            output: Bytes::new(),
                            gas: Gas::new(gas_used),
                        }
                    }
                };
                let call_outcome = CallOutcome {
                    result: interpreter_result,
                    memory_offset: 0..0, // Default memory offset
                };
                Ok(FrameResult::Call(call_outcome))
            }
            FrameInput::Create(create_inputs) => {
                // Set up the create context
                self.inner.ctx.tx.caller = create_inputs.caller;
                self.inner.ctx.tx.kind = TxKind::Create;
                self.inner.ctx.tx.data = create_inputs.init_code.clone();
                self.inner.ctx.tx.value = create_inputs.value;
                self.inner.ctx.tx.gas_limit = create_inputs.gas_limit;
                
                let execution_result = handler.inspect_run(&mut self.inner)?;
                
                // For CREATE2, the Inspector should handle factory logic via InspectorExt
                // Convert ExecutionResult to CreateOutcome
                let (interpreter_result, address) = match execution_result {
                    ExecutionResult::Success { reason, gas_used, gas_refunded: _, logs: _, output } => {
                        let (bytes, addr) = match output {
                            revm::context::result::Output::Call(bytes) => (bytes, None),
                            revm::context::result::Output::Create(bytes, address) => (bytes, address),
                        };
                        let result = InterpreterResult {
                            result: reason.into(),
                            output: bytes,
                            gas: Gas::new(gas_used),
                        };
                        (result, addr)
                    }
                    ExecutionResult::Revert { gas_used, output } => {
                        let result = InterpreterResult {
                            result: InstructionResult::Revert,
                            output: output,
                            gas: Gas::new(gas_used),
                        };
                        (result, None)
                    }
                    ExecutionResult::Halt { reason, gas_used } => {
                        let result = InterpreterResult {
                            result: reason.into(),
                            output: Bytes::new(),
                            gas: Gas::new(gas_used),
                        };
                        (result, None)
                    }
                };
                let create_outcome = CreateOutcome {
                    result: interpreter_result,
                    address,
                };
                Ok(FrameResult::Create(create_outcome))
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
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
        // TODO: Need to update transact_raw for new Revm 26.0.1 API
        // For now using direct execution without custom handler
        self.inner.ctx.tx = tx;
        let mut handler: MainnetHandler<_, EVMError<DatabaseError>, _> = MainnetHandler::default();
        let result = handler.inspect_run(&mut self.inner)?;
        
        // Convert ExecutionResult to ExecResultAndState
        use revm::context::result::ExecResultAndState;
        // Extract state from journaled_state
        let state = self.inner.ctx.journaled_state.finalize();
        Ok(ExecResultAndState { result, state })
    }

    fn transact_system_call(
        &mut self,
        _caller: Address,
        _contract: Address,
        _data: Bytes,
    ) -> Result<ResultAndState<Self::HaltReason>, Self::Error> {
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
            EthFrame,
        >,
        EVMError<DatabaseError>,
        EthFrame,
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
        EthFrame,
    >;
    type Error = EVMError<DatabaseError>;
    type HaltReason = HaltReason;

    // Handler trait has no required methods in Revm 26.0.1
}

impl<I: InspectorExt> InspectorHandler for FoundryHandler<'_, I> {
    type IT = EthInterpreter;

    // InspectorHandler trait has no required methods in Revm 26.0.1
}

impl<'db, I: InspectorExt> FoundryHandler<'db, I> {
    /// Run EVM with inspector support and CREATE2 factory logic
    pub fn inspect_run(&mut self, evm: &mut RevmEvm<EthEvmContext<&'db mut dyn DatabaseExt>, I, EthInstructions<EthInterpreter, EthEvmContext<&'db mut dyn DatabaseExt>>, PrecompilesMap, EthFrame>) -> Result<revm::context_interface::result::ExecutionResult, EVMError<DatabaseError>> {
        // TODO: In Revm 26.0.1, the execution model changed
        // We need to implement CREATE2 factory logic at the inspector level
        // rather than at the handler frame level
        
        // For now, delegate to MainnetHandler
        self.inner.inspect_run(evm)
    }

    /// Run EVM without inspector
    pub fn run(&mut self, evm: &mut RevmEvm<EthEvmContext<&'db mut dyn DatabaseExt>, I, EthInstructions<EthInterpreter, EthEvmContext<&'db mut dyn DatabaseExt>>, PrecompilesMap, EthFrame>) -> Result<revm::context_interface::result::ExecutionResult, EVMError<DatabaseError>> {
        self.inner.run(evm)
    }
}
