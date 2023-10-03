# add directory 
import sys
sys.path.append('../')

# import packages 
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms
from open_cge import calibrate
from open_cge import simpleCGE as cge

# load social accounting matrix
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, 'SAM1.csv')
sam = pd.read_csv(sam_path, index_col=0, header=0, encoding ='latin', on_bad_lines='skip')

# declare sets
u = ('AGR', 'OIL', 'IND', 'SER', 'LAB', 'CAP', 'LAND', 'NTR',
     'DTX', 'IDT', 'ACT', 'HOH', 'GOV', 'INV', 'EXT')
industry = ('AGR', 'OIL', 'IND', 'SER')
h = ('LAB', 'CAP', 'LAND', 'NTR')
w = ('LAB', 'LAND', 'NTR')


def check_square():
    '''
    this function tests whether the SAM is a square matrix.
    '''
    sam_small = sam
    # sam_small = sam_small.drop("TOTAL")
    sam_small.to_numpy(dtype=None, copy=True)
    if not sam_small.shape[0] == sam_small.shape[1]:
        raise ValueError(f"SAM is not square. It has {sam_small.shape[0]} rows and {sam_small.shape[0]} columns")

def row_total():
    '''
    this function tests whether the row sums
    of the SAM equal the expected value.
    '''
    sam_small = sam
    # sam_small = sam_small.drop("TOTAL")
    row_sum = sam_small.sum(axis=0)
    row_sum = pd.Series(row_sum)
    return row_sum

def col_total():
    '''
    this function tests whether column sums
    of the SAM equal the expected values.
    '''
    sam_small = sam
    # sam_small = sam_small.drop("TOTAL")
    col_sum = sam_small.sum(axis=1)
    col_sum = pd.Series(col_sum)
    return col_sum

def row_col_equal():
    '''
    this function tests whether row sums
    and column sums of the SAM are equal.
    '''
    sam_small = sam
    # sam_small = sam_small.drop("TOTAL")
    row_sum = sam_small.sum(axis=0)
    col_sum = sam_small.sum(axis=1)
    np.testing.assert_allclose(row_sum, col_sum)

def runner():
    '''
    this function runs the cge model
    '''

    # solve cge_system
    dist = 10
    tpi_iter = 0
    tpi_max_iter = 1000
    tpi_tol = 1e-10
    xi = 0.1

    # pvec = pvec_init
    pvec = np.ones(len(industry) + len(h))

    # Load data and parameters classes
    data = calibrate.model_data(sam, h, u, industry)
    parameters = calibrate.parameters(data, industry)

    R = data.interest_rate_0
    er = 1

    Zbar = data.output_0
    Ffbar = data.factor_endowment_0
    Kdbar = data.capital_domestic_0
    Qbar = data.domestic_supply_0
    pdbar = pvec[0:len(industry)]

    pm = firms.eqpm(er, data.import_price_index)
    
    i=0
    while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
        print(f'this time is {i}')
        tpi_iter += 1
        cge_args = [parameters, data, industry, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er]

        print("initial guess = ", pvec)
        results = opt.root(cge.cge_system, pvec, args=cge_args, method='lm',
                           tol=1e-5)
        pprime = results.x
        pyprime = pprime[0:len(industry)]
        pfprime = pprime[len(industry):len(industry) + len(h)]
        pyprime = Series(pyprime, index=list(industry))
        pfprime = Series(pfprime, index=list(h))

        pvec = pprime

        pe = firms.eqpe(er, data.export_price_index)
        pm = firms.eqpm(er, data.import_price_index)
        pq = firms.eqpq(pm, pdbar, parameters.taum, parameters.eta, 
                        parameters.deltam, parameters.deltad, parameters.gamma)
        pz = firms.eqpz(parameters.ay, parameters.ax, pyprime, pq)
        Kk = agg.eqKk(pfprime, Ffbar, R, parameters.lam, pq)
        Td = gov.eqTd(parameters.taud, pfprime, Ffbar)
        Trf = gov.eqTrf(parameters.tautr, pfprime, Ffbar)
        Kf = agg.eqKf(Kk, Kdbar)
        Fsh = firms.eqFsh(R, Kf, er)
        Sp = agg.eqSp(parameters.ssp, pfprime, Ffbar, Fsh, Trf)
        I = hh.eqI(pfprime, Ffbar, Sp, Td, Fsh, Trf)
        E = firms.eqE(parameters.theta, parameters.xie, 
                      parameters.production_tax_rate, parameters.phi, pz, pe, Zbar)
        D = firms.eqDex(parameters.theta, parameters.xid, parameters.production_tax_rate, 
                        parameters.phi, pz, pdbar, Zbar)
        M = firms.eqM(parameters.gamma, parameters.deltam, parameters.eta, Qbar, pq, pm, parameters.taum)
        Qprime = firms.eqQ(parameters.gamma, parameters.deltam, parameters.deltad, parameters.eta, M, D)
        pdprime = firms.eqpd(parameters.gamma, parameters.deltam, parameters.eta, Qprime, pq, D)
        Zprime = firms.eqZ(parameters.theta, parameters.xie, parameters.xid, parameters.phi, E, D)
            # Zprime = Zprime.iloc[0]
        Kdprime = agg.eqKd(data.growth_rate, Sp, parameters.lam, pq)
        Ffprime = data.factor_endowment_0
        Ffprime['CAP'] = R * Kk * (parameters.lam * pq).sum() / pfprime[1]

        # dist = (((Zbar - Zprime) ** 2) ** (1 / 2)).sum()
        print('Distance at iteration ', tpi_iter, ' is ', dist)
        pdbar = xi * pdprime + (1 - xi) * pdbar
        Zbar = xi * Zprime + (1 - xi) * Zbar
        Kdbar = xi * Kdprime + (1 - xi) * Kdbar
        Qbar = xi * Qprime + (1 - xi) * Qbar
        Ffbar = xi * Ffprime + (1 - xi) * Ffbar

        Q = firms.eqQ(parameters.gamma, parameters.deltam, parameters.deltad, parameters.eta, M, D)
        i = i +1 
    print('Model solved, Q = ', Q)
    return Q


if __name__ == "__main__":
    check_square()
    row_col_equal()
    runner()
