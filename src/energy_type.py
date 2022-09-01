from enum import Enum

class EnergyType(Enum):
    """
    Represents the different type of renewable energy sources in pypsa-eur
    """
    OFFWIND_AC = "offwind-ac"
    OFFWIND_DC = "offwind-dc"
    ONWIND = "onwind-dc"
    SOLAR = "solar"
    ROR = "ror"
    NOT_DEFINED = "not_defined"

    def get_energy_type(name: str):
        """
        Returns the energy type for a given string
        :param name: energy type as string
        :return: energy type for the given string
        """
        match name:
            case "offwind-ac":
                return EnergyType.OFFWIND_AC
            case "offwind-dc":
                return EnergyType.OFFWIND_DC
            case "onwind":
                return EnergyType.ONWIND
            case "solar":
                return EnergyType.SOLAR
            case "ror":
                return EnergyType.ROR
            case _:
                return EnergyType.NOT_DEFINED
