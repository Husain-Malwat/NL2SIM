#!/usr/bin/env python3

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.mumax3_to_ir import script_to_ir
from validation.ir_to_mumax3 import ir_to_script
from validation.schema_validator import validate_ir


def test_minimal_script():
    """Test 1: Minimal script with basic components."""
    print("\n" + "="*60)
    print("Test 1: Minimal Script")
    print("="*60)
    
    original = """\
SetGridsize(128, 64, 1)
SetCellsize(4e-9, 4e-9, 10e-9)

Msat  = 800e3
Aex   = 13e-12
alpha = 0.02

m = uniform(1, 0.1, 0)
relax()
"""
    
    print("Original script:")
    print(original)
    
    # Parse to IR
    ir = script_to_ir(original)
    print("\n✓ Parsed to IR")
    
    # Validate IR
    valid, errors = validate_ir(ir)
    if not valid:
        print(f"✗ IR validation failed:")
        for err in errors:
            print(f"  {err}")
        return False
    print("✓ IR validates")
    
    # Generate back to script
    regenerated = ir_to_script(ir, add_comments=False)
    print("\nRegenerated script:")
    print(regenerated)
    
    # Check semantic equivalence (key lines should be present)
    checks = [
        ("SetGridsize(128, 64, 1)" in regenerated, "Grid size"),
        ("Msat  = 800000" in regenerated or "Msat  = 8e+05" in regenerated, "Msat"),
        ("Aex   = 1.3e-11" in regenerated, "Aex"),
        ("alpha = 0.02" in regenerated, "alpha"),
        ("uniform(1, 0.1, 0)" in regenerated, "Initial config"),
        ("relax()" in regenerated, "Solver"),
    ]
    
    all_passed = True
    for passed, name in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_vortex_dynamics():
    """Test 2: Vortex with dynamics and output."""
    print("\n" + "="*60)
    print("Test 2: Vortex Dynamics")
    print("="*60)
    
    original = """\
SetGridsize(128, 128, 1)
SetCellsize(4e-9, 4e-9, 10e-9)

Msat  = 860e3
Aex   = 13e-12
alpha = 0.02

m = vortex(1, -1)
relax()

autosave(m, 100e-12)
tableautosave(10e-12)

B_ext = vector(0.01, 0, 0)
run(1e-9)
"""
    
    ir = script_to_ir(original)
    valid, errors = validate_ir(ir)
    
    if not valid:
        print("✗ IR validation failed")
        return False
    
    regenerated = ir_to_script(ir, add_comments=False)
    
    checks = [
        ("vortex(1, -1)" in regenerated, "Vortex config"),
        ("relax()" in regenerated, "Relax"),
        ("autosave(" in regenerated, "Autosave"),
        ("B_ext =" in regenerated, "B_ext"),
        ("run(1e-09)" in regenerated or "run(1e-9)" in regenerated, "Run dynamics"),
    ]
    
    all_passed = True
    for passed, name in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_official_example():
    """Test 3: Parse an actual official example."""
    print("\n" + "="*60)
    print("Test 3: Official Example (Standard Problem #4)")
    print("="*60)
    
    example_path = Path("data/raw/official_examples/example01_std_problem4.mx3")
    
    if not example_path.exists():
        print("⚠️  Example file not found, skipping")
        return True
    
    original = example_path.read_text()
    print(f"Loaded {example_path.name} ({len(original)} bytes)")
    
    # Parse to IR
    ir = script_to_ir(original)
    print("✓ Parsed to IR")
    
    # Validate
    valid, errors = validate_ir(ir)
    if errors:
        print("Validation warnings:")
        for err in errors[:5]:  # Show first 5
            print(f"  {err}")
    
    # Regenerate
    regenerated = ir_to_script(ir, add_comments=True)
    
    # Basic checks
    checks = [
        ("SetGridsize" in regenerated, "Mesh"),
        ("Msat" in regenerated, "Materials"),
        ("m =" in regenerated, "Initial config"),
        ("B_ext" in regenerated or "relax" in regenerated.lower(), "Excitation or solver"),
    ]
    
    all_passed = True
    for passed, name in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_regions():
    """Test 4: Multi-region script."""
    print("\n" + "="*60)
    print("Test 4: Multi-Region Script")
    print("="*60)
    
    original = """\
SetGridsize(128, 128, 1)
SetCellsize(4e-9, 4e-9, 10e-9)

defregion(1, xrange(0, inf))
defregion(2, xrange(-inf, 0))

Msat = 800e3
Aex  = 13e-12
alpha = 0.02

Ku1.SetRegion(1, 1e6)
Ku1.SetRegion(2, 2e6)

m.SetRegion(1, uniform(1, 0, 0))
m.SetRegion(2, uniform(-1, 0, 0))

relax()
"""
    
    ir = script_to_ir(original)
    valid, errors = validate_ir(ir)
    
    if not valid:
        print("✗ IR validation failed")
        return False
    
    regenerated = ir_to_script(ir, add_comments=False)
    
    checks = [
        ("defregion(1," in regenerated, "Region 1"),
        ("defregion(2," in regenerated, "Region 2"),
        ("SetRegion(1," in regenerated or "setregion(1," in regenerated.lower(), "Region params"),
    ]
    
    all_passed = True
    for passed, name in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_stt():
    """Test 5: Spin-transfer torque with current."""
    print("\n" + "="*60)
    print("Test 5: Spin-Transfer Torque")
    print("="*60)
    
    original = """\
SetGridsize(64, 32, 1)
SetCellsize(4e-9, 4e-9, 10e-9)

Msat  = 800e3
Aex   = 13e-12
alpha = 0.02

Pol = 0.56
xi  = 0.1

m = uniform(1, 0, 0)
relax()

J = vector(1e12, 0, 0)
run(1e-9)
"""
    
    ir = script_to_ir(original)
    valid, errors = validate_ir(ir)
    
    if not valid:
        print("✗ IR validation failed")
        return False
    
    regenerated = ir_to_script(ir, add_comments=False)
    
    checks = [
        ("J = vector" in regenerated, "Current density"),
        ("Pol =" in regenerated, "Polarization"),
        ("xi =" in regenerated, "Non-adiabaticity"),
    ]
    
    all_passed = True
    for passed, name in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all roundtrip tests."""
    print("\n" + "#"*60)
    print("# IR CONVERTER ROUNDTRIP TESTS")
    print("#"*60)
    
    tests = [
        ("Minimal Script", test_minimal_script),
        ("Vortex Dynamics", test_vortex_dynamics),
        ("Official Example", test_official_example),
        ("Multi-Region", test_regions),
        ("Spin-Transfer Torque", test_stt),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed_count} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
