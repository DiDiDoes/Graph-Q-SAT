from pysat.formula import CNF
import torch
from torch_geometric.data import Data

from minisat_wrapper import MiniSAT

class CNFLoader:
    def __init__(self, cnf_file: str):
        self.cnf_file = cnf_file

    def load_cnf(self):
        cnf = CNF(from_file=self.cnf_file)
        clauses = [list(clause) for clause in cnf.clauses]
        clauses = [clause for clause in clauses if clause]  # Filter out empty clauses
        self.clauses = clauses
        self.num_variables = cnf.nv
        self.num_clauses = len(clauses)

    def build_vcg(self):
        # Building the Variable-Clause Graph (VCG)
        # Deprecated: we will get the VCG directly from the MiniSAT wrapper in main.py, but this is how we would build it ourselves if needed.

        # Node features: [1, 0] for variable nodes, [0, 1] for clause nodes
        x = [[1, 0]] * self.num_variables + [[0, 1]] * self.num_clauses

        # Edge features: [0, 1] for positive literals, [1, 0] for negative literals
        variable_indices = []
        clause_indices = []
        edge_attr = []
        for i_clause, clause in enumerate(self.clauses):
            for literal in clause:
                variable = abs(literal) - 1
                variable_indices.append(variable)
                clause_indices.append(i_clause + self.num_variables)
                edge_attr.append([0, 1] if literal > 0 else [1, 0])

        # Create the PyTorch Geometric Data object
        edge_index = [variable_indices, clause_indices]
        self.graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        )

def build_vcg_from_solver(solver: MiniSAT, device: torch.device) -> Data:
    x, edge_index, edge_attr = solver.get_vcg()
    graph = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )
    graph.to(device)
    return graph

if __name__ == "__main__":
    # Example usage
    cnf_file = "data/uf50-218/uf50-01.cnf"
    loader = CNFLoader(cnf_file)
    loader.load_cnf()
    print("Number of variables:", loader.num_variables)
    print("Number of clauses:", loader.num_clauses)

    # Access the graph data
    loader.build_vcg()
    graph_data = loader.graph
    print(graph_data)

    # Run the MiniSAT wrapper
    from minisat_wrapper import MiniSAT
    solver = MiniSAT(cnf=loader.clauses)
