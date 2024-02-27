from cobra.io import read_sbml_model
from gurobipy import *

class Simulator():
    """
    Simulator for flux balance analysis
    """
    def __init__(self):
        self.cobra_model = None
        self.model_metabolites = None
        self.model_reactions = None
        self.model_genes = None
        self.Smatrix = None
        self.lower_boundary_constraints = None
        self.upper_boundary_constraints = None
        self.objective = None

    def read_model(self, filename):
        model = read_sbml_model(filename)
        self.cobra_model = model

        model_metabolites = [each_metabolite.id for each_metabolite in model.metabolites]
        model_reactions = []
        model_genes = [each_gene.id for each_gene in model.genes]

        Smatrix = {}

        lower_boundary_constraints = {}
        upper_boundary_constraints = {}

        objective_reaction = ''

        for each_reaction in model.reactions:
            if each_reaction.objective_coefficient == 1.0:
                objective_reaction = each_reaction.id

            reactant_list = each_reaction.reactants
            reactant_coff_list = list(each_reaction.get_coefficients(reactant_list))

            product_list = each_reaction.products
            product_coff_list = list(each_reaction.get_coefficients(product_list))

            for i in range(len(reactant_list)):
                Smatrix[(reactant_list[i].id, each_reaction.id)] = reactant_coff_list[i]

            for i in range(len(product_list)):
                Smatrix[(product_list[i].id, each_reaction.id)] = product_coff_list[i]

            model_reactions.append(each_reaction.id)

            lb = each_reaction.lower_bound
            ub = each_reaction.upper_bound
            if lb < -1000.0:
                lb = float('-inf')
            if ub > 1000.0:
                ub = float('inf')
            lower_boundary_constraints[each_reaction.id] = lb
            upper_boundary_constraints[each_reaction.id] = ub

        self.model_metabolites = model_metabolites
        self.model_reactions = model_reactions
        self.model_genes = model_genes
        self.Smatrix = Smatrix
        self.lower_boundary_constraints = lower_boundary_constraints
        self.upper_boundary_constraints = upper_boundary_constraints
        self.objective = objective_reaction

        return (model_metabolites, model_reactions, Smatrix,
                lower_boundary_constraints, upper_boundary_constraints, objective_reaction)

    def run_FBA(self, new_objective='', flux_constraints={}, inf_flag=False, internal_flux_minimization=False, mode='max'):
        lower_boundary_constraints = self.lower_boundary_constraints
        upper_boundary_constraints = self.upper_boundary_constraints

        if not inf_flag:
            for key in lower_boundary_constraints:
                if lower_boundary_constraints[key] == float("-inf"):
                    lower_boundary_constraints[key] = -1000.0

            for key in upper_boundary_constraints:
                if upper_boundary_constraints[key] == float("inf"):
                    upper_boundary_constraints[key] = 1000.0

        pairs, coffvalue = multidict(self.Smatrix)

        m = Model('FBA')
        m.setParam('OutputFlag', 0)
        m.reset()

        m.params.Threads = 1
        m.update()

        v = {}
        fplus = {}
        fminus = {}

        m.update()

        for each_reaction in self.model_reactions:
            if each_reaction in flux_constraints:
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], name=each_reaction)
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction], ub=upper_boundary_constraints[each_reaction],
                                            name=each_reaction)
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)
        m.update()

        for each_metabolite in self.model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            else:
                m.addConstr(quicksum(v[reaction] * coffvalue[metabolite, reaction] for metabolite, reaction in
                                 pairs.select(each_metabolite, '*')) == 0)
        m.update()

        if new_objective == '':
            objective = self.objective
        else:
            objective = new_objective

        if mode == 'max':
            m.setObjective(v[objective], GRB.MAXIMIZE)
        else:
            m.setObjective(v[objective], GRB.MINIMIZE)

        m.optimize()

        if m.status == 2:
            objective_value = m.ObjVal

            if internal_flux_minimization:
                # 是否进行内部通量最小化（pFBA）
                m.addConstr(fplus[objective] - fminus[objective] == objective_value)
                # for each_metabolite in model_metabolites:???
                # fplus：正向通量 fminus：负向通量
                m.addConstr(quicksum(
                    (fplus[reaction] - fminus[reaction]) * coffvalue[metabolite, reaction] for metabolite, reaction in
                    pairs.select(each_metabolite, '*')) == 0)

                for each_reaction in self.model_reactions:
                    m.addConstr(fplus[each_reaction] - fminus[each_reaction] == v[each_reaction])

                m.update()
                m.setObjective(
                    quicksum((fplus[each_reaction] + fminus[each_reaction]) for each_reaction in self.model_reactions),
                    GRB.MINIMIZE)
                m.optimize()

                if m.status == 2:
                    objective_value = m.ObjVal
                    flux_distribution = {}
                    for reaction in self.model_reactions:
                        flux_distribution[reaction] = v[reaction].x
                    return m.status, objective_value, flux_distribution

            else:
                flux_distribution = {}
                for reaction in self.model_reactions:
                    flux_distribution[reaction] = v[reaction].x
                    if abs(float(v[reaction].x)) <= 1e-6: # 1e-6?
                        flux_distribution[reaction] = 0.0
                return m.status, objective_value, flux_distribution
        return m.status, False, False

    def run_MOMA(self, wild_flux={}, flux_constraints={}, inf_flag=False):
        lower_boundary_constraints = self.lower_boundary_constraints
        upper_boundary_constraints = self.upper_boundary_constraints

        if not inf_flag:
            for key in lower_boundary_constraints:
                if lower_boundary_constraints[key] == float("-inf"):
                    lower_boundary_constraints[key] = -1000.0

            for key in upper_boundary_constraints:
                if upper_boundary_constraints[key] == float("inf"):
                    upper_boundary_constraints[key] = 1000.0

        pairs, coffvalue = multidict(self.Smatrix)

        m = Model('MOMA')
        m.setParam('OutputFlag', 0)
        m.reset()

        m.params.Threads = 1
        m.update()

        wild_flux_of_reactions = wild_flux.keys()
        v = {}
        fplus = {}
        fminus = {}

        for each_reaction in self.model_reactions:
            if each_reaction in flux_constraints:
                v[each_reaction] = m.addVar(lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1], name=each_reaction)
            else:
                v[each_reaction] = m.addVar(lb=lower_boundary_constraints[each_reaction], ub=upper_boundary_constraints[each_reaction],
                                            name=each_reaction)
            fplus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)
            fminus[each_reaction] = m.addVar(lb=0.0, ub=1000.0, name=each_reaction)
        m.update()

        for each_reaction in self.model_reactions:
            m.addConstr(v[each_reaction] == (fplus[each_reaction] - fminus[each_reaction]))
            m.addConstr(fplus[each_reaction], GRB.GREATER_EQUAL, v[each_reaction] - wild_flux[each_reaction],
                        name=each_reaction)
            m.addConstr(fminus[each_reaction], GRB.GREATER_EQUAL, wild_flux[each_reaction] - v[each_reaction],
                        name=each_reaction)
        m.update()

        for each_metabolite in self.model_metabolites:
            if len(pairs.select(each_metabolite, '*')) == 0:
                continue
            m.addConstr(quicksum((fplus[reaction] - fminus[reaction]) * coffvalue[metabolite, reaction]
                                 for metabolite, reaction in pairs.select(each_metabolite, '*')) == 0)
        m.update()

        m.setObjective(quicksum(
            ((fplus[each_reaction] + fminus[each_reaction]) - wild_flux[each_reaction]) for each_reaction in
            wild_flux_of_reactions), GRB.MINIMIZE)

        m.optimize()

        if m.status == 2:
            flux_distribution = {}
            for reaction in self.model_reactions:
                flux_distribution[reaction] = float(v[reaction].x)
            return m.status, m.ObjVal, flux_distribution
        else:
            return m.status, False, False
