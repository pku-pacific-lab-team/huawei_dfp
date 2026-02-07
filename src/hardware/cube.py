class CubeSpec:
    def __init__(
        self,
        m_dim: int = 16,
        n_dim: int = 16,
        k_dim: int = 16,
        p_input: int = 16,
    ):
        self.m_dim = m_dim
        self.n_dim = n_dim
        self.k_dim = k_dim
        self.p_input = p_input

    @classmethod 
    def from_dict(cls, config: dict) -> "CubeSpec":
        return cls(
            m_dim=config.get("m_dim", 16),
            n_dim=config.get("n_dim", 16),
            k_dim=config.get("k_dim", 16),
            p_input=config.get("p_input", 16),
        )

    def __repr__(self):
        return f"CubeSpec(m_dim={self.m_dim}, n_dim={self.n_dim}, k_dim={self.k_dim}, p_input={self.p_input})"