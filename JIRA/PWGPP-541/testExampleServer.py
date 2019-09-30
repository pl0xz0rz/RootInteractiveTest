import numpy as np

from bokeh.layouts import row, widgetbox
from bokeh.models import Slider
from bokeh.plotting import figure, ColumnDataSource, curdoc


def foo(xval, a, b):
    print(xval.min(), xval.max(), np.isnan(xval).any(), a, b)
    return np.power(xval, a) + b


# some artificial data
a0 = 2.
b0 = 1.
x = np.linspace(-100., 100, 1000)
y = foo(x, a0, b0)

source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

a_slider_obj = Slider(start=0, end=3, value=a0, step=0.1, id='a_slider', title="a_slider")
b_slider_obj = Slider(start=-4, end=4, value=b0, step=0.5, id='b_slider', title="b_slider")


def callback(attr, old, new):
    data = source.data
    # Since the callback is used by two sliders, we can't just use the `new` argument
    a_dynamic = a_slider_obj.value
    b_dynamic = b_slider_obj.value

    # Here I assume that you wanted to change the value and not just create an unused variable
    data['y'] = foo(data['x'], a_dynamic, b_dynamic)


a_slider_obj.on_change('value', callback)
b_slider_obj.on_change('value', callback)

layout = row(
    plot,
    widgetbox(a_slider_obj, b_slider_obj),
)

curdoc().add_root(layout)