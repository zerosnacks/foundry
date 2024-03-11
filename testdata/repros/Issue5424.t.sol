// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity 0.8.18;

import "ds-test/test.sol";
import "../cheats/Vm.sol";

// https://github.com/foundry-rs/foundry/issues/5424
contract Issue5424Test is DSTest {
    Vm constant vm = Vm(HEVM_ADDRESS);

    struct SimpleDynamic {
        uint256[] timestamp;
    }

    function testParseJsonDynamicArray() public {
        string memory json = '{"timestamp": [1655140035]}';
        bytes memory data = vm.parseJson(json);
        SimpleDynamic memory fixture = abi.decode(data, (SimpleDynamic));
        assertEq(fixture.timestamp[0], 1655140035);
    }

    function testParseTomlDynamicArray() public {
        string memory toml = "timestamp = [1655140035]";
        bytes memory data = vm.parseToml(toml);
        SimpleDynamic memory fixture = abi.decode(data, (SimpleDynamic));
        assertEq(fixture.timestamp[0], 1655140035);
    }

    // struct SimpleStatic {
    //     uint256[1] timestamp;
    // }

    uint256[1] public timestamp;

    function testParseJsonStaticArray() public {
        // string memory json = '{"timestamp": [1655140035]}';
        // bytes memory data = vm.parseJson(json);
        // bytes memory data = "0x7b2274696d657374616d70223a205b313635353134303033355d";
        // SimpleStatic memory fixture = abi.decode(data, (SimpleStatic));
        // assertEq(fixture.timestamp[0], 1655140035);
    }

    function testParseTomlStaticArray() public {
        string memory toml = "timestamp = [1655140035]";
        timestamp[0] = vm.parseJsonUintArray(toml, ".timestamp");

        assertEq(timestamp[0], 1655140035);

        // bytes memory data = vm.parseToml(toml);
        // timestamp = abi.decode(data, (uint256[1]));
    }
}
