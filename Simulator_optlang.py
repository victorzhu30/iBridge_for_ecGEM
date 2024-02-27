"""
Gurobi:statu：2
optlang:status:optimal
"""

from optlang import Model, Variable, Constraint, Objective
from cobra.io import read_sbml_model

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

        pairs = list(self.Smatrix.keys())
        coffvalue = self.Smatrix

        # 创建一个模型
        m = Model("FBA")

        # 创建变量并添加到模型中
        v = {}
        fplus = {}
        fminus = {}
        for each_reaction in self.model_reactions:
            if each_reaction in flux_constraints:
                v[each_reaction] = Variable(name='v'+each_reaction, lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1])
            else:
                v[each_reaction] = Variable(name='v' + each_reaction, lb=lower_boundary_constraints[each_reaction],
                                            ub=upper_boundary_constraints[each_reaction])
                fplus[each_reaction] = Variable(name='fplus' + each_reaction, lb=0.0, ub=1000.0)
                fminus[each_reaction] = Variable(name='fminus' + each_reaction, lb=0.0, ub=1000.0)
                m.add([v[each_reaction], fplus[each_reaction], fminus[each_reaction]])

        # 添加约束条件
        for each_metabolite in self.model_metabolites:
            each_metabolite_pairs_list = [pair for pair in pairs if each_metabolite in pair]
            if len(each_metabolite_pairs_list) != 0:
                sum1 = 0
                for pair in each_metabolite_pairs_list:
                    reaction = pair[1]
                    metabolite = pair[0]
                    sum1 += v[reaction] * coffvalue[metabolite, reaction]
                m.add([Constraint(sum1, lb=0, ub=0)])

        # 设置目标函数
        if new_objective == '':
            objective = self.objective
        else:
            objective = new_objective

        if mode == 'max':
            obj1 = Objective(v[objective], direction='max')
            m.objective = obj1
        else:
            obj1 = Objective(v[objective], direction='min')
            m.objective = obj1

        # 求解模型
        m.optimize()

        if m.status == 'optimal':
            objective_value = m.objective.value

            if internal_flux_minimization:
                # 是否进行内部通量最小化（pFBA）
                m.add([Constraint(fplus[objective] - fminus[objective] - objective_value, lb=0, ub=0)])
                for each_metabolite in self.model_metabolites:
                    each_metabolite_pairs_list = [pair for pair in pairs if each_metabolite in pair]
                    sum2 = 0
                    for pair in each_metabolite_pairs_list:
                        reaction = pair[1]
                        metabolite = pair[0]
                        sum2 += (fplus[reaction] - fminus[reaction]) * coffvalue[metabolite, reaction]
                    m.add([Constraint(sum2, lb=0, ub=0)])

                for each_reaction in self.model_reactions:
                    m.add([Constraint(fplus[each_reaction] - fminus[each_reaction] - v[each_reaction], lb=0, ub=0)])

                sum2 = 0
                for each_reaction in self.model_reactions:
                    sum2+= fplus[each_reaction] + fminus[each_reaction]

                obj2 = Objective(sum2,direction='min')

                m.objective = obj2
                m.optimize()

                if m.status == 'optimal':
                    objective_value = m.objective.value
                    flux_distribution = {reaction: v[reaction].primal for reaction in self.model_reactions}
                    return m.status, objective_value, flux_distribution

            else:
                flux_distribution = {reaction: v[reaction].primal for reaction in self.model_reactions}
                for reaction in self.model_reactions:
                    if abs(flux_distribution[reaction]) <= 1e-6:  # 1e-6?
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

        pairs = list(self.Smatrix.keys())
        coffvalue = self.Smatrix

        m = Model('MOMA')

        wild_flux_of_reactions = wild_flux.keys()
        v = {}
        fplus = {}
        fminus = {}

        for each_reaction in self.model_reactions:
            if each_reaction in flux_constraints:
                v[each_reaction] = Variable(name='v'+each_reaction, lb=flux_constraints[each_reaction][0],
                                            ub=flux_constraints[each_reaction][1])
            else:
                v[each_reaction] = Variable(name='v'+each_reaction,lb=lower_boundary_constraints[each_reaction],
                                            ub=upper_boundary_constraints[each_reaction])
            fplus[each_reaction] = Variable(name='fplus'+each_reaction, lb=0.0, ub=1000.0)
            fminus[each_reaction] = Variable(name='fminus'+each_reaction,lb=0.0, ub=1000.0, )
            m.add([v[each_reaction], fplus[each_reaction], fminus[each_reaction]])

        for each_reaction in self.model_reactions:
            m.add([Constraint(v[each_reaction] - (fplus[each_reaction] - fminus[each_reaction]), lb=0, ub=0)])
            m.add([Constraint(v[each_reaction] - wild_flux[each_reaction] - fplus[each_reaction], ub=0)])
            m.add([Constraint(wild_flux[each_reaction] - v[each_reaction] - fminus[each_reaction], ub=0)])

        for each_metabolite in self.model_metabolites:
            each_metabolite_pairs_list = [pair for pair in pairs if each_metabolite in pair]
            if len(each_metabolite_pairs_list) != 0:
                sum3 = 0
                for pair in each_metabolite_pairs_list:
                    reaction = pair[1]
                    metabolite = pair[0]
                    sum3 += (fplus[reaction] - fminus[reaction]) * coffvalue[metabolite, reaction]
                m.add([Constraint(sum3, lb=0, ub=0)])

        obj3 = 0
        for each_reaction in wild_flux_of_reactions:
            obj3 += (fplus[each_reaction] + fminus[each_reaction]) - wild_flux[each_reaction]
        m.objective = Objective(expression=obj3, direction='min')

        m.optimize()

        if m.status == 'optimal':
            objective_value = m.objective.value
            flux_distribution = {}
            for reaction in self.model_reactions:
                flux_distribution[reaction] = v[reaction].primal
            return m.status, objective_value, flux_distribution
        else:
            return m.status, False, False
