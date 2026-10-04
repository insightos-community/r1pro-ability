from r1pro_abilities.entrypoint import run_ability
from r1pro_abilities.service import AbilityRole


if __name__ == "__main__":
    run_ability(AbilityRole.OBJECT_PERCEPTION, "R1ProObjectPerception.V2")
