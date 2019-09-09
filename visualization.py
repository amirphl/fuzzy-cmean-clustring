import numpy as np
from bokeh.plotting import figure, show, output_file

colors = ['#FF6B33', '#9CFF33', '#33FFCA', '#336BFF', '#FF33FF', '#9ecae1', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2',
          '#31a354', '#74c476', '#c7e9c0', '#756bb1', '#bcbddc', '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9']


def draw_model_2d(model, data=None, membership=None, show_figure=True):
    title = "draw FCM model"
    fig = figure(title=title, toolbar_location=None)
    fig.grid.grid_line_color = None
    fig.background_fill_color = "#eeeeee"
    for clus, cc_color in enumerate(zip(model.cluster_centers_, colors)):
        cc, color = cc_color
        fig = draw_points_2d(np.array([cc]), fig=fig, title=title, marker="diamond", size=15,
                             line_color="navy", fill_color=color, alpha=1.0)
        if data is not None and membership is not None:
            for idx, data_point in enumerate(data):
                # print(membership[idx][clus])
                fig = draw_points_2d(np.array([data_point]), fig=fig, title=title, marker="circle", size=10,
                                     line_color="navy", fill_color=color, alpha=membership[idx][clus])
    if show_figure:
        show(fig)
    return fig


def draw_points_2d(points, fig=None, title="figure 123", **kwargs):
    if fig is None:
        fig = figure(title=title, toolbar_location=None)
        fig.grid.grid_line_color = None
        fig.background_fill_color = "#eeeeee"
    x, y = points.T
    fig.scatter(x, y, **kwargs)
    output_file("plot.html", title=title + " of outputfile")
    return fig
