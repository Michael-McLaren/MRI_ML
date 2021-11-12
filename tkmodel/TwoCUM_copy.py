# TwoCUM module containing model function TwoCUM and fitting function TwoCUMfitting
# Also includes tofts without vp to do toff estimation
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import scipy.optimize

def TwoCUM(params,t,AIF, toff):

    #print(params)
    # If toff value needs to be included (i.e. if not set to None), shift the AIF by the amount toff
    # Note when evaluating the model during fitting this shift has already been done
    #plt.plot(AIF)
    if toff != None:
        tnew = t - toff
        f=scipy.interpolate.interp1d(t,AIF,kind='linear',bounds_error=False,fill_value=0)
        AIF = (t>=toff)*f(t-toff)
        #plt.plot(AIF)
    #Test for trouble with fitting algorithm
    if np.isnan(np.sum(params)):
        F=np.zeros(len(AIF))
        return F

    # Assign the parameters to more meaningful names
    E, Fp, vp = params
    # E between 0 and 1
    # FP blood flow
    # vp between 0 and 1

    #First calculate the parameter Tp
    Tp=(vp/Fp)*(1-E)

    #Calculate the IRF
    exptTp=np.exp(-1*t/Tp)

    R=exptTp*(1-E) + E
    #Calculate the convolution
    temp=np.convolve(AIF,R)*t[1]

    F=Fp*temp[0:len(t)]
    #plt.plot(AIF)
    #plt.plot(t/60,R,'x')
    #plt.plot(F)
    return F


# Model function for Kety with no vp to use for toff calculation
def Kety(params,t,AIF):
    #Test for trouble with fitting algorithm
    if np.isnan(np.sum(params)):
        F=np.zeros(len(AIF))
        return F

    # Assign parameter names
    Ktrans, ve, toff = params

    # Shift the AIF by the amount toff
    tnew = t - toff
    f=scipy.interpolate.interp1d(t,AIF,kind='linear',bounds_error=False,fill_value=0)
    AIFnew = (t>toff)*f(t-toff)

    # Shift the AIF by the number of time points closest to the fitted toff
    # Find closest point:
    #toffrnd=np.argmin(abs(t-toff))
    # Shift the AIF
    #AIFnew=np.roll(AIF,toffrnd)
    # Set early points before toff to zero
    #AIFnew = (t>t[toffrnd])*AIFnew

    imp=Ktrans*np.exp(-1*Ktrans*t/ve); # Calculate the impulse response function
    convolution=np.convolve(AIFnew,imp) # Convolve impulse response with AIF
    G=convolution[0:len(t)]*t[1]
    return G


def TwoCUMfittingConc(t, AIF, uptake, toff, plot=0):

    # If toff is set to None, rather than a number, calculate it using Tofts without vp from the first third of the curve
    #plt.figure()

    firstthird=int(np.round(len(t)/3))
    if toff is None:
        Ketystart=np.array((0.01,0.1,t[7])) # set starting guesses for Ktrans, ve, toff
        #Ketystart=Ketystart+[t[1]]
        Ketybnds=((0.00001,10),(0.00001,2),(0.00001,30))
        Ketyresult=scipy.optimize.minimize(KetyobjfunConc,Ketystart,args=(t[0:firstthird],AIF[0:firstthird],uptake[0:firstthird]),bounds=Ketybnds,method='SLSQP',options={'disp':False})
        toff=0
        if not np.isnan(Ketyresult.x[2]):
            toff=Ketyresult.x[2]

        if plot==1:
            plt.figure()
            plt.plot(t,uptake,'rx')
            plt.plot(t[0:firstthird],Kety(Ketyresult.x[0:4],t[0:firstthird],AIF[0:firstthird]))
            print(Ketyresult.x)
            print(Ketyresult.success)

    # Shift the AIF by the amount toff
    tnew = t - toff
    f=scipy.interpolate.interp1d(t,AIF,kind='linear',bounds_error=False,fill_value=0)
    AIFnew = (t>=toff)*f(t-toff)

    # Fit the TwoCXM model, stepping through vp
    vpmatrix=np.arange(0.01,1,0.01) #was 0.01 start
    # Parameters to fit are E, Fp
    startguess=np.array((0.1,0.05))  # Set starting guesses
    bnds=((0,1.2),(0.0000001,10)) # Set upper and lower bounds for parameters
    resultsmatrix=np.zeros((len(vpmatrix),6))  # Initialise results array

    for i in range (0,len(vpmatrix)):
        Result=scipy.optimize.minimize(objfunConc,startguess,args=(np.array([vpmatrix[i]]),t,AIFnew,uptake),bounds=bnds,options={'ftol':1e-14,'disp':False,'maxiter':300})
        #print(Result.x,vpmatrix[i],Result.fun,Result.success)
        resultsmatrix[i,:]=(Result.x[0],Result.x[1],vpmatrix[i],Result.fun,toff,Result.status)


    bestindex=np.nanargmin(resultsmatrix[:,3])
    bestresult=resultsmatrix[bestindex,:]
    #print(bestresult)
    if plot==1:
        print(bestresult)
        plt.figure()
        plt.plot(t,uptake,'rx')
        plt.plot(t,TwoCUM(bestresult[0:3],t,AIF,toff),'k-',linewidth=4)
    return bestresult
    # E, fp, vp , result of least square, toff, and if its a good fit, 0 if works, 1 if problem

def KetyobjfunConc(paramsin,t,AIF,data):
    #print(paramsin)
    temp=data-Kety(paramsin,t,AIF)
    return np.sqrt(np.sum(temp**2))


def objfunConc(paramsin,vp,t,AIF,data):
    allparams=np.concatenate((paramsin,vp))
    temp=data-TwoCUM(allparams,t,AIF,None)
    return np.sqrt(np.sum(temp**2))





# could fit an offset time
