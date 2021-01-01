import numpy as np
import plotly.graph_objects as go

x = np.linspace(0, 500, 500)
y1 = 0.5*np.sin(2*np.pi*x)
y2 = 0.2*np.sin(2*np.pi*x*0.99)
y3 = y1+y2
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=x, y=y1, mode='lines'))
fig1.add_trace(go.Scatter(x=x, y=y2, mode='lines'))
fig1.add_trace(go.Scatter(x=x, y=y3, mode='lines'))
fig1.update_layout(xaxis_title='time', yaxis_title='amplitude', title='Waves')
fig1.show()


