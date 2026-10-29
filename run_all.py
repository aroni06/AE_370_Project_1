"""Run every verification study and the main experiment, in report order."""

from studies import (mass_experiment, timestep_study, verify_conservation, verify_rebound,
                     verify_two_body)

for name, mod in (("Verification 1: exact two-body limit", verify_two_body),
                  ("Verification 2: conservation laws", verify_conservation),
                  ("Verification 3: REBOUND IAS15 cross-check", verify_rebound),
                  ("Time-step study", timestep_study),
                  ("Mass experiment", mass_experiment)):
    print(f"\n=== {name} ===")
    mod.main()
