import numbers

import numpy
from AnyQt.QtWidgets import QFormLayout, QSizePolicy, QHeaderView
from AnyQt.QtCore import Qt, QItemSelection

from pyqtgraph import Point

# Maximum number of PCA components that we can set in the widget
from orangewidget.gui import TableView
from orangewidget.utils.itemmodels import PyTableModel

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.data.util import get_unique_names
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT
from Orange.preprocess import preprocess
from Orange.projection import PCA
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.slidergraph import SliderGraph
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


MAX_COMPONENTS = 100
LINE_NAMES = ["component variance", "cumulative variance"]


class OWPCA(widget.OWWidget):
    name = "PCA"
    description = "Principal component analysis with a scree-diagram."
    icon = "icons/PCA.svg"
    priority = 3050
    keywords = ["principal component analysis", "linear transformation"]

    class Inputs:
        data = Input("Data", Table)
        test_data = Input("Test Data", Table)

    class Outputs:
        transformed_data = Output("Transformed Data", Table, replaces=["Transformed data"])
        data = Output("Data", Table, default=True)
        components = Output("Components", Table)
        pca = Output("PCA", PCA, dynamic=False)

    ncomponents = settings.Setting(2)
    variance_covered = settings.Setting(100)
    auto_commit = settings.Setting(True)
    normalize = settings.Setting(True)
    maxp = settings.Setting(20)
    axis_labels = settings.Setting(10)

    graph_name = "plot.plotItem"

    class Warning(widget.OWWidget.Warning):
        trivial_components = widget.Msg(
            "All components of the PCA are trivial (explain 0 variance). "
            "Input data is constant (or near constant).")

    class Error(widget.OWWidget.Error):
        no_features = widget.Msg("At least 1 feature is required")
        no_instances = widget.Msg("At least 1 data instance is required")

    def __init__(self):
        super().__init__()
        self.data = None
        self.test_data = None

        self._pca = None
        self._transformed = None
        self._variance_ratio = None
        self._cumulative = None
        self._test_variance = None
        self._init_projector()

        # Options
        self.options_box = gui.vBox(self.controlArea, "Options")
        self.normalize_box = gui.checkBox(
            self.options_box, self, "normalize",
            "Normalize variables", callback=self._update_normalize,
            attribute=Qt.WA_LayoutUsesWidgetRect
        )

        self.maxp_spin = gui.spin(
            self.options_box, self, "maxp", 1, MAX_COMPONENTS,
            label="Show only first", callback=self._maxp_changed,
            keyboardTracking=False
        )

        # Components Selection
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        box = gui.widgetBox(self.controlArea, "Components Selection",
                            orientation=form)

        self.components_spin = gui.spin(
            box, self, "ncomponents", 1, MAX_COMPONENTS,
            callback=self._component_spin_changed,
            keyboardTracking=False, addToLayout=False,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            alignment=Qt.AlignRight
        )
        self.components_spin.setSpecialValueText("All")

        self.variance_spin = gui.spin(
            box, self, "variance_covered", 1, 100,
            callback=self._variance_spin_changed,
            keyboardTracking=False, addToLayout=False,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            alignment=Qt.AlignRight
        )
        self.variance_spin.setSuffix(" %")

        form.addRow("Components:", self.components_spin)
        form.addRow("Explained variance:", self.variance_spin)

        self.varmodel = PyTableModel(parent=self)
        view = self.varview = TableView(
            self,
            sortingEnabled=False,
            selectionMode=TableView.SelectionMode.NoSelection)
        self.varview.clicked.connect(self._select_row)
        view.horizontalHeader().setDefaultAlignment(Qt.AlignRight)
        view.verticalHeader().setVisible(True)
        view.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        view.setModel(self.varmodel)
        form.addRow(view)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

        self.plot = PCASliderGraph(
            "Principal Components", "Proportion of variance",
            self._on_cut_changed)

        self.mainArea.layout().addWidget(self.plot)
        self._update_normalize()

    @Inputs.data
    def set_data(self, data):
        self.clear_messages()
        self.clear()
        self.information()
        self.data = None
        if not data:
            self.clear_outputs()
        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                self.information("Data has been sampled")
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(2000, partial=True)
                data = Table(data_sample)
        if isinstance(data, Table):
            if not data.domain.attributes:
                self.Error.no_features()
                self.clear_outputs()
                return
            if not data:
                self.Error.no_instances()
                self.clear_outputs()
                return

        self._init_projector()
        self.data = data
        self.fit()

    @Inputs.test_data
    def set_test_data(self, test_data):
        self.test_data = test_data

    def handleNewSignals(self):
        self.test_plot_commit()

    def test_plot_commit(self):
        self.run_test_data()
        self._setup_plot()
        self._setup_table()
        self.unconditional_commit()

    def fit(self):
        self.clear()
        self.Warning.trivial_components.clear()
        if self.data is None:
            return

        data = self.data

        if self.normalize:
            self._pca_projector.preprocessors = \
                self._pca_preprocessors + [preprocess.Normalize(center=False)]
        else:
            self._pca_projector.preprocessors = self._pca_preprocessors

        if not isinstance(data, SqlTable):
            pca = self._pca_projector(data)
            variance_ratio = pca.explained_variance_ratio_
            cumulative = numpy.cumsum(variance_ratio)

            if numpy.isfinite(cumulative[-1]):
                self.components_spin.setRange(0, len(cumulative))
                self._pca = pca
                self._variance_ratio = variance_ratio
                self._cumulative = cumulative
            else:
                self.Warning.trivial_components()

    def run_test_data(self):
        if self._pca is None or self.test_data is None:
            self._test_variance = None
            return

        projected = self._pca(self.test_data).X
        self._test_variance = numpy.var(projected, axis=0)
        self._test_variance /= numpy.sum(self._test_variance)

    def clear(self):
        self._pca = None
        self._transformed = None
        self._variance_ratio = None
        self.varmodel.clear()
        self._test_variance = None
        self._cumulative = None
        self.plot.clear_plot()

    def clear_outputs(self):
        self.Outputs.transformed_data.send(None)
        self.Outputs.data.send(None)
        self.Outputs.components.send(None)
        self.Outputs.pca.send(self._pca_projector)

    def _maxp_changed(self):
        self._setup_plot()
        self._setup_table()

    def _setup_table(self):
        if self._pca is None:
            self.varmodel.clear()
            return

        columns = (self._variance_ratio[:self.maxp],
                   numpy.cumsum(self._variance_ratio))
        if self._test_variance is not None:
            columns += (self._test_variance, numpy.cumsum(self._test_variance))
            self.varmodel.setHorizontalHeaderLabels(("Var", "Cumul",
                                                     "Test var", "Cumul"))
        else:
            self.varmodel.setHorizontalHeaderLabels(("Variance", "Cumulative"))

        self.varmodel[:] = zip(*columns)
        # This can't be set in __init__ because columns must exist
        view = self.varview
        for header in (view.horizontalHeader(), view.verticalHeader()):
            header.setSectionResizeMode(QHeaderView.Stretch)
        self._update_table_selection()

    def _select_row(self, index):
        self.ncomponents = index.row() + 1
        self._component_spin_changed()

    def _update_table_selection(self):
        index = self.varmodel.index
        selmodel = self.varview.selectionModel()
        selmodel.select(
            QItemSelection(index(0, 0), index(self.ncomponents - 1, 3)),
            selmodel.ClearAndSelect)

    def _setup_plot(self):
        if self._pca is None:
            self.plot.clear_plot()
            return

        explained_ratio = self._variance_ratio
        explained = self._cumulative
        cutpos = self._nselected_components()
        p = min(len(self._variance_ratio), self.maxp)
        xs = numpy.arange(1, p + 1)

        yss = [explained_ratio[:p], explained[:p]]
        colors = [Qt.red, Qt.darkYellow]
        widths = [2, 2]
        if self._test_variance is not None:
            yss += [self._test_variance[:p],
                    numpy.cumsum(self._test_variance[:p])]
            colors *= 2
            widths += [4, 4]

        self.plot.update(
            xs, yss, colors, cutpoint_x=cutpos, names=LINE_NAMES, widths=widths)
        self._update_axis()

    def _on_cut_changed(self, components):
        if components == self.ncomponents \
                or self.ncomponents == 0:
            return

        self.ncomponents = components
        if self._pca is not None:
            var = self._cumulative[components - 1]
            if numpy.isfinite(var):
                self.variance_covered = int(var * 100)

        self._update_table_selection()

        self._invalidate_selection()

    def _component_spin_changed(self):
        # cut changed by "ncomponents" spin.
        if self._pca is None:
            self._invalidate_selection()
            return

        if self.ncomponents == 0:
            # Special "All" value
            cut = len(self._variance_ratio)
        else:
            cut = self.ncomponents

        var = self._cumulative[cut - 1]
        if numpy.isfinite(var):
            self.variance_covered = int(var * 100)

        self.plot.set_cut_point(cut)
        self._update_table_selection()
        self._invalidate_selection()

    def _variance_spin_changed(self):
        # cut changed by "max variance" spin.
        if self._pca is None:
            return

        cut = numpy.searchsorted(self._cumulative,
                                 self.variance_covered / 100.0) + 1
        cut = min(cut, len(self._cumulative))
        self.ncomponents = cut
        self._update_table_selection()
        self.plot.set_cut_point(cut)
        self._invalidate_selection()

    def _update_normalize(self):
        self.fit()
        self.test_plot_commit()
        if self.data is None:
            self._invalidate_selection()

    def _init_projector(self):
        self._pca_projector = PCA(n_components=MAX_COMPONENTS, random_state=0)
        self._pca_projector.component = self.ncomponents
        self._pca_preprocessors = PCA.preprocessors

    def _nselected_components(self):
        """Return the number of selected components."""
        if self._pca is None:
            return 0

        if self.ncomponents == 0:
            # Special "All" value
            max_comp = len(self._variance_ratio)
        else:
            max_comp = self.ncomponents

        var_max = self._cumulative[max_comp - 1]
        if var_max != numpy.floor(self.variance_covered / 100.0):
            cut = max_comp
            assert numpy.isfinite(var_max)
            self.variance_covered = int(var_max * 100)
        else:
            self.ncomponents = cut = numpy.searchsorted(
                self._cumulative, self.variance_covered / 100.0) + 1
        return cut

    def _invalidate_selection(self):
        self.commit()

    def _update_axis(self):
        p = min(len(self._variance_ratio), self.maxp)
        axis = self.plot.getAxis("bottom")
        d = max((p-1)//(self.axis_labels-1), 1)
        axis.setTicks([[(i, str(i)) for i in range(1, p + 1, d)]])

    def commit(self):
        transformed = data = components = None
        if self._pca is not None:
            if self._transformed is None:
                # Compute the full transform (MAX_COMPONENTS components) once.
                self._transformed = self._pca(self.data)
            transformed = self._transformed

            if self._variance_ratio is not None:
                for var, explvar in zip(
                        transformed.domain.attributes,
                        self._variance_ratio[:self.ncomponents]):
                    var.attributes["variance"] = round(explvar, 6)
            domain = Domain(
                transformed.domain.attributes[:self.ncomponents],
                self.data.domain.class_vars,
                self.data.domain.metas
            )
            transformed = transformed.from_table(domain, transformed)

            # prevent caching new features by defining compute_value
            proposed = [a.name for a in self._pca.orig_domain.attributes]
            meta_name = get_unique_names(proposed, 'components')
            meta_vars = [StringVariable(name=meta_name)]
            metas = numpy.array([['PC{}'.format(i + 1)
                                  for i in range(self.ncomponents)]],
                                dtype=object).T
            if self._variance_ratio is not None:
                variance_name = get_unique_names(proposed, "variance")
                meta_vars.append(ContinuousVariable(variance_name))
                metas = numpy.hstack(
                    (metas,
                     self._variance_ratio[:self.ncomponents, None]))

            dom = Domain(
                [ContinuousVariable(name, compute_value=lambda _: None)
                 for name in proposed],
                metas=meta_vars)
            components = Table(dom, self._pca.components_[:self.ncomponents],
                               metas=metas)
            components.name = 'components'

            data_dom = Domain(
                self.data.domain.attributes,
                self.data.domain.class_vars,
                self.data.domain.metas + domain.attributes)
            data = Table.from_numpy(
                data_dom, self.data.X, self.data.Y,
                numpy.hstack((self.data.metas, transformed.X)),
                ids=self.data.ids)

        self._pca_projector.component = self.ncomponents
        self.Outputs.transformed_data.send(transformed)
        self.Outputs.components.send(components)
        self.Outputs.data.send(data)
        self.Outputs.pca.send(self._pca_projector)

    def send_report(self):
        if self.data is None:
            return
        self.report_items((
            ("Normalize data", str(self.normalize)),
            ("Selected components", self.ncomponents),
            ("Explained variance", "{:.3f} %".format(self.variance_covered))
        ))
        self.report_plot()
        self.report_table("Variances per component", self.varview)

    @classmethod
    def migrate_settings(cls, settings, version):
        if "variance_covered" in settings:
            # Due to the error in gh-1896 the variance_covered was persisted
            # as a NaN value, causing a TypeError in the widgets `__init__`.
            vc = settings["variance_covered"]
            if isinstance(vc, numbers.Real):
                if numpy.isfinite(vc):
                    vc = int(vc)
                else:
                    vc = 100
                settings["variance_covered"] = vc
        if settings.get("ncomponents", 0) > MAX_COMPONENTS:
            settings["ncomponents"] = MAX_COMPONENTS

        # Remove old `decomposition_idx` when SVD was still included
        settings.pop("decomposition_idx", None)

        # Remove RemotePCA settings
        settings.pop("batch_size", None)
        settings.pop("address", None)
        settings.pop("auto_update", None)


class PCASliderGraph(SliderGraph):
    def _update_horizontal_lines(self):
        # When showing four curves, in case the labels on the train and test
        # curve overlap, but one below and one above the curve
        super()._update_horizontal_lines()
        if len(self.sequences) != 4:
            return

        for ci in (0, 1):
            lab0, lab1 = (self.plot_horlabel[ci + i] for i in (0, 2))
            if not lab0.textItem.collidesWithItem(lab1.textItem):
                continue

            first_up = lab0.pos().y() > lab1.pos().y()
            lab0.anchor = Point(lab0.anchor[0], first_up)
            lab0.updateTextPos()
            lab1.anchor = Point(lab1.anchor[0], not first_up)
            lab1.updateTextPos()


if __name__ == "__main__":  # pragma: no cover
    data = Table("housing")
    test = numpy.zeros(len(data), dtype=bool)
    test[::3] = True
    WidgetPreview(OWPCA).run(set_data=data[:-30],
                             set_test_data=data[-50:])
