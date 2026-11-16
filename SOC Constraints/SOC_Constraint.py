
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
# Replace the original SOC initialization:
#
# soc = np.random.normal(loc=0.5, scale=0.16, size=None)
#
# With:
#
# soc = apply_soc_constraints(
#     np.random.normal(loc=0.5, scale=0.16)
# )
# -------------------------------------------------


# -------------------------------------------------
# Modification 2: Replace _update_agent_state()
# in the original Price class with:
# -------------------------------------------------

def _update_agent_state(self, agent, action):

    # Update EV SOC
    self.soc_list[agent] += action

    # Apply SOC constraints
    self.soc_list[agent] = apply_soc_constraints(
        self.soc_list[agent]
    )
