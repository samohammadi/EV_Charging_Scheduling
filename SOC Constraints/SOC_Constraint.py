# SOC Constraints
# Minimum SOC = 20%
# Maximum SOC = 100%

MIN_SOC = 0.20
MAX_SOC = 1.00


def apply_soc_constraints(soc):
    """
    Restrict battery SOC between 20% and 100%.
    """
    return np.clip(soc, MIN_SOC, MAX_SOC)


# -------------------------------------------------
# Modification 1: In reset()
# -------------------------------------------------

soc = apply_soc_constraints(
    np.random.normal(loc=0.5, scale=0.16)
)


# -------------------------------------------------
# Modification 2: Replace _update_agent_state()
# -------------------------------------------------

def _update_agent_state(self, agent, action):

    delta_t = 1.0              # hour
    battery_capacity = 24.0   # kWh

    # Previous SOC
    old_soc = self.soc_list[agent]

    # Convert MW to kWh
    energy_change = action * 1000 * delta_t

    # Convert energy change to SOC change
    soc_change = energy_change / battery_capacity

    # Update EV SOC
    new_soc = old_soc + soc_change

    # Apply SOC constraints
    self.soc_list[agent] = apply_soc_constraints(new_soc)
