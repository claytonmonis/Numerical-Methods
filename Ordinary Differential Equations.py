import matplotlib.pyplot
import inspect

def func(t, y):
    return y + t

def rk4(t0, y0, h = 1, start = 0, stop = 10):
    assert start < stop
    
    yList = []
    tList = []
    
    for i in range((stop - start)//h + 1):
        tList.append(t0)
        yList.append(y0)

        if len(inspect.signature(func).parameters) == 1:
            k1 = h * func(y0)
            k2 = h * func(y0 + k1/2.0)
            k3 = h * func(y0 + k2/2.0)
            k4 = h * func(y0 + k3)
        elif len(inspect.signature(func).parameters) == 2:
            k1 = h * func(t0, y0)
            k2 = h * func(t0 + h/2.0, y0 + k1/2.0)
            k3 = h * func(t0 + h/2.0, y0 + k2/2.0)
            k4 = h * func(t0 + h, y0 + k3)

        y0 = y0 +(k1 + 2.0 * k2 + 2.0 * k3 + k4)/6.0
        t0 += h
    
    matplotlib.pyplot.plot(tList,yList,'r')
    return tList, yList

def heun(t0, y0, h = 1, start = 0, stop = 10):
    assert start < stop
    
    yList = []
    tList = []
    
    for i in range((stop - start)//h + 1):
        tList.append(t0)
        yList.append(y0)

        if len(inspect.signature(func).parameters) == 1:
            yhat = h * func(y0)
            y0 += (h / 2) * (func(y0) + func(yhat))
        elif len(inspect.signature(func).parameters) == 2:
            yhat = h * func(t0, y0)
            y0 += (h / 2) * (func(t0, y0) + func(t0 + h, yhat))
        
        t0 += h
    
    matplotlib.pyplot.plot(tList,yList,'r')
    return tList, yList

def euler(t0, y0, h = 1, start = 0, stop = 10):
    assert start < stop
    
    yList = []
    tList = []
    
    for i in range((stop - start)//h + 1):
        tList.append(t0)
        yList.append(y0)

        if len(inspect.signature(func).parameters) == 1:
            y0 += h * func(y0)
        elif len(inspect.signature(func).parameters) == 2:
            y0 += h * func(t0, y0)

        t0 += h
    
    matplotlib.pyplot.plot(tList,yList,'r')
    return tList, yList
