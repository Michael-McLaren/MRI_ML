

def plot_PK(output, true):
    #input should be numpy arrays with 3 values
    AIF_plot = np.load("../data/AIF.npy")
    t_plot = np.arange(0, 366, 2.45)
    fitted_curve = TwoCUM(output, t_plot, AIF_plot, 0)
    plt.plot(t_plot, fitted_curve, label='Prediction')
    fitted_curve = TwoCUM(true, t_plot, AIF_plot, 0)
    plt.plot(t_plot, fitted_curve, label='True')
    plt.legend()
    plt.show()