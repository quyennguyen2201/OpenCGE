# import packages
import numpy as np
from pandas import Series, DataFrame


class model_data(object):
    '''
    This function reads the SAM file and initializes variables using
    these data.

    Args:
        sam (DataFrame): DataFrame containing social and economic data

    Returns:
        model_data (data class): Data used in the CGE model
    '''

    def __init__(self, sam, h, u, industry):
        # foreign saving
        self.saving_foreign_0 = DataFrame(sam, index=['INV'], columns=['EXT'])
        # private saving
        self.saving_private_0 = DataFrame(sam, index=['INV'], columns=['HOH'])
        # government saving/budget balance
        self.saving_government_0 = DataFrame(sam, index=['INV'], columns=['GOV'])
        # repatriation of profits
        self.profit_repatriation_0 = DataFrame(sam, index=['EXT'], columns=['HOH'])
        # capital stock
        self.capital_0 = 10510
        # foreign-owned capital stock
        self.capital_foreign_0 = 6414.35
        # domestically-owned capital stock
        self.capital_domestic_0 = self.capital_0 - self.capital_foreign_0

        # direct tax
        self.tax_direct0 = DataFrame(sam, index=['DTX'], columns=['HOH'])
        # transfers
        self.transfer_0 = DataFrame(sam, index=['HOH'], columns=['GOV'])
        # production tax
        self.tax_production_0 = DataFrame(sam, index=['ACT'], columns=list(industry))
         # import tariff
        self.tariff_0 = DataFrame(sam, index=['IDT'], columns=list(industry))

        # the h-th factor input by the j-th firm
        self.factor_0 = DataFrame(sam, index=list(h), columns=list(industry))
        # factor endowment of the h-th factor
        self.factor_endowment_0 = self.factor_0.sum(axis=1)
        # composite factor (value added)
        self.composite_factor_0 = self.factor_0.sum(axis=0)
        # intermediate input
        self.intermediate_input_0 = DataFrame(sam, index=list(industry), columns=list(industry))
        # total intermediate input by the j-th sector
        self.total_intermediate_input_0 = self.intermediate_input_0.sum(axis=0)
        # output of the i-th good
        self.output_0 = self.composite_factor_0 + self.total_intermediate_input_0

        # household consumption of the i-th good
        self.consumption_household_0 = DataFrame(sam, index=list(industry), columns=['HOH'])
        # government consumption
        self.consumption_government_0 = DataFrame(sam, index=list(industry), columns=['GOV'])
        # investment demand
        self.investment_demand_0 = DataFrame(sam, index=list(industry), columns=['INV'])
        # exports
        self.export_0 = DataFrame(sam, index=list(industry), columns=['EXT'])
        self.export_0 = self.export_0['EXT']
        # imports
        self.import_0 = DataFrame(sam, index=['EXT'], columns=list(industry))
        self.import_0 = self.import_0.loc['EXT']

        # domestic supply/Armington composite good
        self.domestic_supply_0 = (self.consumption_household_0['HOH'] +
                  self.consumption_government_0['GOV'] + self.investment_demand_0['INV']
                   + self.intermediate_input_0.sum(axis=1))
        # production tax rate
        self.production_tax_rate = self.tax_production_0 / self.output_0
        # domestic tax rate
        self.domestic_tax_rate = (1 + self.production_tax_rate.loc['ACT']) * self.output_0 - self.export_0

        # Compute aggregates

        # aggregate output
        self.total_output_0 = self.output_0.sum()
        # aggregate demand
        self.total_consumption_household_0 = self.consumption_household_0.sum()
        # aggregate investment
        self.total_investment_demand_0 = self.investment_demand_0.sum()
        # aggregate government spending
        self.total_consumption_government_0 = self.consumption_government_0.sum()
        # aggregate imports
        self.total_import_0 = self.import_0.sum()
        # aggregate exports
        self.total_export_0 = self.export_0.sum()
        # aggregate gross domestic product
        self.Gdp0 = (self.total_output_0 + self.total_consumption_household_0 
                     + self.total_consumption_government_0 + self.total_export_0 -
                     self.total_import_0)
        # growth rate of capital stock
        self.growth_rate = self.total_investment_demand_0 / self.capital_0
        # interest rate
        self.interest_rate_0 = self.factor_endowment_0['CAP'] / self.capital_0

        # export price index
        self.export_price_index  = np.ones(len(industry))
        self.export_price_index = Series(self.export_price_index, index=list(industry))
        # import price index
        self.import_price_index  = np.ones(len(industry))
        self.import_price_index = Series(self.import_price_index, index=list(industry))


class parameters(object):
    '''
    This function sets the values of parameters used in the model.

    Args:

    Returns:
        parameters (parameters class): Class of parameters for use in
            CGE model.
    '''

    def __init__(self, data, industry):

        # elasticity of substitution
        self.sigma = ([3, 1.2, 3, 3])
        self.sigma = Series(self.sigma, index=list(industry))
        # substitution elasticity parameter
        self.eta = (self.sigma - 1) / self.sigma

        # elasticity of transformation
        self.psi = ([3, 1.2, 3, 3])
        self.psi = Series(self.psi, index=list(industry))
        # transformation elasticity parameter
        self.phi = (self.psi + 1) / self.psi

        # share parameter in utility function
        self.alpha = data.consumption_household_0 / data.total_consumption_household_0
        self.alpha = self.alpha['HOH']
        # share parameter in production function
        self.beta = data.factor_0 / data.composite_factor_0
        temp = data.factor_0 ** self.beta
        # scale parameter in production function
        self.b = data.composite_factor_0 / temp.prod(axis=0)

        # intermediate input requirement coefficient
        self.ax = data.intermediate_input_0 / data.output_0
        # composite factor input requirement coefficient
        self.ay = data.composite_factor_0 / data.output_0
        self.mu = data.consumption_government_0 / data.total_consumption_government_0
        # government consumption share
        self.mu = self.mu['GOV']
        self.lam = data.investment_demand_0 / data.total_investment_demand_0
        # investment demand share
        self.lam = self.lam['INV']

        # production tax rate
        self.production_tax_rate = data.total_output_0 / data.output_0
        # self.production_tax_rate = self.production_tax_rate.loc['ACT']
        # import tariff rate
        self.taum = data.total_import_0 / data.import_0
        # self.taum = self.taum.loc['IDT']

        # share parameter in Armington function
        self.deltam = ((1 + self.taum) * data.import_0 ** (1 - self.eta) /
                       ((1 + self.taum) * data.import_0 ** (1 - self.eta) + data.domestic_tax_rate
                        ** (1 - self.eta)))
        self.deltad = (data.domestic_tax_rate ** (1 - self.eta) /
                       ((1 + self.taum) * data.import_0 ** (1 - self.eta) + data.domestic_tax_rate
                        ** (1 - self.eta)))

        # scale parameter in Armington function
        self.gamma = (data.domestic_supply_0 / (self.deltam * data.import_0 ** self.eta +
                              self.deltad * data.domestic_tax_rate ** self.eta) **
                      (1 / self.eta))

        # share parameter in transformation function
        self.xie = (data.total_export_0 ** (1 - self.phi) / (data.total_export_0 ** (1 - self.phi) +
                                              data.domestic_tax_rate ** (1 - self.phi)))
        self.xid = (data.domestic_tax_rate ** (1 - self.phi) / (data.total_export_0 ** (1 - self.phi) +
                                              data.domestic_tax_rate ** (1 - self.phi)))

        # scale parameter in transformation function
        self.theta = (data.output_0 / (self.xie * data.total_export_0 ** self.phi + self.xid *
                              data.domestic_tax_rate ** self.phi) ** (1 / self.phi))

        # average propensity to save
        self.ssp = (data.saving_private_0.values / (data.factor_endowment_0.sum() - data.profit_repatriation_0.values +
                                    data.transfer_0.values))
        self.ssp = self.ssp
        # direct tax rate
        self.taud = data.tax_direct0.values / data.factor_endowment_0.sum()
        self.taud = np.array(self.taud)
        # transfer rate
        self.tautr = data.transfer_0.values / data.factor_endowment_0['LAB']
        self.tautr = np.array(self.tautr)
        # government revenue
        self.ginc = data.tax_direct0  + data.tax_production_0.sum() + data.total_import_0.sum()
        # household income
        self.hinc = data.factor_endowment_0.sum()
