import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


ee = 1.60217663e-19
me = 9.1093837e-31
mp = 1.67262192e-27
eps0 = 8.85418782e-12
mu0 = 1.25663706e-6


def calc_ne_from_nuei_and_beta(inp, ai):
    knob = 4.0 / 3.0
    ne_unit = 1.0e20
    te_unit = 1.0e3
    kk = knob * ne_unit * np.power(te_unit, -2.0) * np.sqrt(2.0 * np.pi * ee / me) / np.power(4.0 * np.pi * eps0 / ee, 2.0)
    #kk = (1.0e4 / 1.09) / np.sqrt(te_unit)
    cb = inp['betae'] * np.power(inp['bref'], 2.0) / (2.0 * mu0 * ee * ne_unit * te_unit)
    cnu = inp['nuei'] * np.power(cb, 2.0) / (kk * np.sqrt(ai * mp / ee) * inp['lref'] * inp['zeff'])
    data = {'logterm': np.log(cb.to_numpy()), 'constant': cnu.to_numpy()}
    rootdata = pd.DataFrame(data)
    func_ne20 = lambda row: root_scalar(
        lambda ne: (15.2 + row['logterm'] - 1.5 * np.log(ne)) * np.power(ne, 3.0) - row['constant'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: (15.2 + row['logterm'] - 1.5 * np.log(ne)) * np.power(ne, 3.0) - row['constant'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    ne_data = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    ne = pd.DataFrame(data=ne_data, index=inp.index)
    return ne


def compute_normalized_chie_bohm(inp):
    const = 2.5e-4
    agrad = inp['eps'] * np.abs(inp['ane'] + inp['ate'])
    chie_bohm_norm = const * np.power(inp['q'], 2.0) * agrad
    return chie_bohm_norm


def compute_normalized_chii_bohm(inp):
    const = 2.5e-4
    agrad = inp['eps'] * np.abs(inp['ani'] + inp['ati'])
    chii_bohm_norm = const * np.power(inp['q'], 2.0) * agrad
    return chii_bohm_norm


def compute_normalized_chie_gyrobohm(inp):
    const = 3.2e-1
    agrad = inp['eps'] * np.abs(inp['ane'] + inp['ate'])
    rgrad = inp['x'] * np.abs(inp['ane'] + inp['ate'])
    chie_gyrobohm_norm = const * np.power(np.abs(inp['q']), 3.0) * np.sqrt(inp['betai'] + inp['betae']) * agrad * rgrad
    return chie_gyrobohm_norm


def compute_normalized_chii_gyrobohm(inp):
    const = 3.5e-2
    agrad = inp['eps'] * np.abs(inp['ate'])
    chii_gyrobohm_norm = const * agrad
    return chii_gyrobohm_norm


def predict(inp):
    '''
    Mixed 97 Bohm/gyro-Bohm (BgB) model
    Taken from M. Erba et al., Nuclear Fusion 38 (1998)

    Requires Pandas DataFrame input containing as columns:
        lref,    in m (probably a)
        bref,    in T
        betae,   normalized with bref
        nuei,    normalized with cs/a
        ate,     normalized with a
        ati,     normalized with a
        ane,     normalized with a
        tinorm,  as Ti/Te
        ninorm,  as ni/ne
        q,
        zeff,
        eps,     as a/Rmaj
        x        as rmin_local/a
    '''
    outp = copy.deepcopy(inp)

    ai = 2.5    # Assumes 50/50 deuterium/tritium

    outp['nref'] = calc_ne_from_nuei_and_beta(outp, ai)
    outp['tref'] = outp['betae'] * np.power(outp['bref'], 2.0) / (2.0 * mu0 * ee * outp['nref'])

    rho_s = np.sqrt(ai * mp * outp['tref'] / ee) / outp['bref']
    norm_factor = outp['lref'] / rho_s

    outp['betai'] = outp['betae'] * outp['ninorm'] * outp['tinorm']
    outp['ani'] = outp['ane'].copy()

    chie_n_b = compute_normalized_chie_bohm(outp)
    chie_n_gb = compute_normalized_chie_gyrobohm(outp)
    chii_n_b = compute_normalized_chii_bohm(outp)
    chii_n_gb = compute_normalized_chii_gyrobohm(outp)

    outp['chie_bgb'] = 0.5 * chie_n_b * norm_factor + chie_n_gb
    outp['chii_bgb'] = 2.0 * chii_n_b * norm_factor + chii_n_gb
    outp['de_bgb'] = np.abs(1.0 - 0.3 * outp['x']) * outp['chie_bgb'] * outp['chii_bgb'] / (outp['chie_bgb'] + outp['chii_bgb'])
    outp['di_bgb'] = outp['de_bgb'].copy()

    # Assumes pure diffusion
    outp['qe_bgb'] = outp['chie_bgb'] * outp['ate']
    outp['qi_bgb'] = outp['chii_bgb'] * outp['ati']
    outp['gammae_bgb'] = outp['de_bgb'] * outp['ane']
    outp['gammai_bgb'] = outp['di_bgb'] * outp['ani']

    return outp


def parse_inputs():
    desc =  'Computes anomalous transport coefficients according to the mixed 97 Bohm/gyro-Bohm model.\n\n'
    desc += 'Input DataFrame must have the following column names and units:\n\n'
    desc += '  lref   - in m\n'
    desc += '  bref   - in T\n'
    desc += '  betae  - normalized with bref\n'
    desc += '  nuei   - normalized with cs/a\n'
    desc += '  ate    - normalized with a\n'
    desc += '  ati    - normalized with a\n'
    desc += '  ane    - normalized with a\n'
    desc += '  tinorm - normalized with te\n'
    desc += '  ninorm - normalized with ne\n'
    desc += '  q      - \n'
    desc += '  zeff   - \n'
    desc += '  eps    - normalized as a/Rmaj\n'
    desc += '  x      - normalized as rmin_local/a\n\n'
    desc += 'See M. Erba et al., Nuclear Fusion 38 (1998) for more details.'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument('ifile', type=str, help='Path to HDF5 file containing Pandas DataFrame of input parameters')
    parser.add_argument('ofile', type=str, help='Path to HDF5 file to store output parameters')
    parser.add_argument('--ikey', type=str, default=None, help='Custom key inside input HDF5 file to access Pandas DataFrame of input parameters')
    parser.add_argument('--okey', type=str, default=None, help='Custom key inside output HDF5 file to access Pandas DataFrame of output parameters')
    args = parser.parse_args()
    return args


def main():
    args = parse_inputs()
    ipath = Path(args.ifile)
    if ipath.is_file():
        opath = Path(args.ofile)
        if not opath.parent.exists():
            opath.parent.mkdir(parents=True)
        ikey = args.ikey if args.ikey is not None else '/data'
        okey = args.okey if args.okey is not None else '/data'
        inp = pd.read_hdf(ipath, key=ikey)
        outp = predict(inp)
        outp.to_hdf(opath, key=okey)
    print('Script completed!')


if __name__ == "__main__":
    main()

