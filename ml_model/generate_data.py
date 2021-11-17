import numpy as np

def generate_xy(num_curves):
    AIF = np.load("../data/AIF.npy")
    data_size = AIF.shape[0]
    t = np.arange(0, 366, 2.45)

    E = np.random.rand(1, num_curves)  # 0 to 1 for both E and vp
    vp = np.random.rand(1, num_curves)
    Fp = 1e-5 * np.random.rand(1, num_curves)

    #     Fp = abs(np.random.normal(size=num_curves, loc= 1e-5, scale = 1e-4)[None,:])

    E_Fp = np.concatenate((E, Fp), axis=0)
    y = np.concatenate((E_Fp, vp), axis=0)

    x = np.zeros((num_curves, data_size))
    for i in range(num_curves):
        x[i] = TwoCUM(y[:, i], t, AIF, 0)

    y = y.T

    return x, y

def main():

    x, y