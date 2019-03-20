from IPython.core.display import display
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.glyphs import Quad
from bokeh.io import push_notebook
from ipywidgets import *
import pyparsing
from itertools import izip


class drawHisto(object):
    def __init__(self, histograms, selection, **options):

        self.selectionList = parseSelectionString(selection)
        self.histArray = histograms
        self.sliderList = []
        self.sliderNames = []
        self.initSlider("")
        WidgetBox = widgets.VBox(self.sliderList)

        self.figure, self.handle, self.source = self.drawGraph(self.histArray, self.selectionList, **options)
        self.updateInteractive("")
        display(WidgetBox)

    def updateInteractive(self, b):
        for hisTitle, projectionList in izip(*[iter(self.selectionList)] * 2):
            for iDim in range(0, self.histArray.FindObject(hisTitle).GetNdimensions() - 1):
                iSlider = self.sliderNames.index(self.histArray.FindObject(hisTitle).GetAxis(iDim).GetTitle())
                value = self.sliderList[iSlider].value
                self.histArray.FindObject(hisTitle).GetAxis(iDim).SetRangeUser(value[0], value[1])
        iterator = 0
        for hisTitle, projectionList in izip(*[iter(self.selectionList)] * 2):
            dimList = map(int, projectionList)
            nDim = len(dimList)
            if nDim > 1:
                raise NotImplementedError("Sorry!!.. Multidimensional projections have not been implemented, yet")
            histogram = self.histArray.FindObject(hisTitle).Projection(dimList[0])
            binsLowEdge = []
            binsUpEdge = []
            top = []
            bottom = []
            for i in range(1, histogram.GetNbinsX() + 1):
                binsLowEdge.append(histogram.GetXaxis().GetBinLowEdge(i))
                binsUpEdge.append(histogram.GetXaxis().GetBinUpEdge(i))
                top.append(histogram.GetBinContent(i))
                bottom.append(0)
            newSource = ColumnDataSource(dict(
                left=binsLowEdge,
                right=binsUpEdge,
                top=top,
                bottom=bottom
            ))
            self.source[iterator].data = newSource.data
            iterator = iterator + 1
        push_notebook(self.handle)

    def initSlider(self, b):
        for hisTitle, projectionList in izip(*[iter(self.selectionList)] * 2):
            for iDim in range(0, self.histArray.FindObject(hisTitle).GetNdimensions() - 1):
                axis = self.histArray.FindObject(hisTitle).GetAxis(iDim)
                title = axis.GetTitle()
                maxRange = axis.GetXmax()
                minRange = axis.GetXmin()
                nBin = axis.GetNbins()
                step = (maxRange - minRange) / nBin
                slider = makeSlider(title, minRange, maxRange, step)
                if title not in self.sliderNames:
                    slider.observe(self.updateInteractive, names='value')
                    self.sliderList.append(slider)
                    self.sliderNames.append(title)

    def drawGraph(self, hisArray, selectionList, **options):
        if 'ncols' in options.keys():
            nCols = options['ncols']
        else:
            nCols = 2
        p = []
        source = []
        iterator = 0
        for hisTitle, projectionList in izip(*[iter(selectionList)] * 2):
            dimList = map(int, projectionList)
            nDim = len(dimList)
            if nDim > 1:
                raise NotImplementedError("Sorry!!.. Multidimensional projections have not been implemented, yet")
            histogram = hisArray.FindObject(hisTitle).Projection(dimList[0])
            binsLowEdge = []
            binsUpEdge = []
            top = []
            bottom = []
            for i in range(1, histogram.GetNbinsX() + 1):
                binsLowEdge.append(histogram.GetXaxis().GetBinLowEdge(i))
                binsUpEdge.append(histogram.GetXaxis().GetBinUpEdge(i))
                top.append(histogram.GetBinContent(i))
                bottom.append(0)
            histLabel = histogram.GetTitle()
            xLabel = histogram.GetXaxis().GetTitle()
            yLabel = histogram.GetYaxis().GetTitle()
            source.append(ColumnDataSource(dict(
                left=binsLowEdge,
                right=binsUpEdge,
                top=top,
                bottom=bottom
            )
            )
            )
            tools = 'pan,box_zoom, wheel_zoom,box_select,lasso_select,reset'
            #   localHist = make_plot(histLabel, top, binsLowEdge, binsUpEdge, xLabel, yLabel, **options)
            localHist = figure(title=histLabel, tools=tools, background_fill_color="#fafafa")
            glyph = Quad(top="top", bottom="bottom", left="left", right="right", fill_color="navy", line_color="white")
            localHist.add_glyph(source[iterator], glyph)
            localHist.y_range.start = 0
            localHist.xaxis.axis_label = xLabel
            localHist.yaxis.axis_label = yLabel
            p.append(localHist)
            iterator = iterator + 1
        pAll = gridplot(p, ncols=nCols, plot_width=400, plot_height=400)
        handle = show(pAll, notebook_handle=True)
        return pAll, handle, source


def makeSlider(title, minRange, maxRange, step):
    slider = widgets.FloatRangeSlider(description=title, layout=Layout(width='66%'), min=minRange, max=maxRange,
                                      step=step, value=[minRange, maxRange])
    return slider


def parseSelectionString(selectionString):
    toParse = "(" + selectionString + ")"
    theContent = pyparsing.Word(pyparsing.alphanums + ".+-") | '#' | pyparsing.Suppress(',')
    selectionParser = pyparsing.nestedExpr('(', ')', content=theContent)
    selectionList = selectionParser.parseString(toParse)[0]
    return selectionList
