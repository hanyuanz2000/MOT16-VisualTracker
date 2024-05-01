from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit as st
import streamlit.components.v1 as components

# Helper function to create a bar chart
def create_bar_chart(data, title, bar_color = '#5470c6'):
    bar = Bar()
    bar.add_xaxis(list(data.keys()))
    bar.add_yaxis(title, list(data.values()), itemstyle_opts=opts.ItemStyleOpts(color=bar_color))
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),  # Optional: rotate labels if they overlap or are too long
        yaxis_opts=opts.AxisOpts(name='Score')  # Optional: add a label to the Y-axis
    )
    # display x and y axis labels
    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=True))
    return bar.render_embed()