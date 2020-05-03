from typing import ClassVar, Iterator, Optional, Tuple

from pandas import DataFrame, Series
from plotly.graph_objs import Figure, Frame, Scatter
from plotly.subplots import make_subplots


class VecData:
    def __init__(self, df: DataFrame, columns: Tuple[str, str, str], factor: float = 1):
        self.df = df.reset_index(drop=True)
        self.columns = columns
        self.df[list(columns)] *= factor

    @property
    def x(self) -> Series:
        return self.df[self.columns[0]]

    @property
    def y(self) -> Series:
        return self.df[self.columns[1]]

    @property
    def z(self) -> Series:
        return self.df[self.columns[2]]

    @property
    def xyz(self) -> DataFrame:
        return self.df[list(self.columns)]

    def max_range(self, margin: float = 0) -> Tuple[float, float]:
        return (self.xyz.min().min() - margin, self.xyz.max().max() + margin)

    def x_range(self, margin: float = 0) -> Tuple[float, float]:
        return (self.x.min() - margin, self.x.max() + margin)

    def y_range(self, margin: float = 0) -> Tuple[float, float]:
        return (self.y.min() - margin, self.y.max() + margin)

    def z_range(self, margin: float = 0) -> Tuple[float, float]:
        return (self.z.min() - margin, self.z.max() + margin)

    @property
    def triples(self) -> Iterator[Tuple[float, float, float]]:
        for x, y, z in self.df.itertuples(index=False):
            yield x, y, z

    def append(self, xyz: Tuple[float, float, float]) -> None:
        self.df = self.df.append(
            {col: val for col, val in zip(self.columns, xyz)}, ignore_index=True,
        )

    @staticmethod
    def range_magnitude(r: Tuple[float, float]) -> float:
        return abs(r.__getitem__(1) - r.__getitem__(0))


class Plotter:
    POINT_SIZE: ClassVar[int] = 20
    VERTICAL_SPACING: ClassVar[float] = 0.07
    ACCELERATION_UNIT: ClassVar[str] = " m/s²"
    TIME_UNIT: ClassVar[str] = " s"
    POS_COLUMNS: ClassVar[Tuple[str, str, str]] = ("pos_x", "pos_y", "pos_z")

    def __init__(self, acc: VecData, frequency: float):
        self.acc = acc
        self.frequency = frequency
        self.pos = self.generate_position_df(self.frequency)

    def make_line_plot(self, action: str) -> Figure:
        self.figure = make_subplots(
            rows=3,
            cols=2,
            shared_xaxes=True,
            subplot_titles=(
                [
                    "X Acceleration",
                    "X Distance",
                    "Y Acceleration",
                    "Y Distance",
                    "Z Acceleration",
                    "Z Distance",
                ]
            ),
            vertical_spacing=self.VERTICAL_SPACING,
            specs=[[{}, {}], [{}, {}], [{}, {}]],
        )

        self.figure.layout.title = (
            f"{round(len(self.acc.df) / self.frequency, 1)} Seconds of {action}"
        )

        self.figure.add_trace(
            Scatter(
                x=self.acc.df.index.to_series() / self.frequency,
                y=self.acc.x,
                mode="lines",
                name="X Acceleration",
            ),
            row=1,
            col=1,
        )
        self.figure.add_trace(
            Scatter(
                x=self.acc.df.index.to_series() / self.frequency,
                y=self.acc.y,
                mode="lines",
                name="Y Acceleration",
            ),
            row=2,
            col=1,
        )
        self.figure.add_trace(
            Scatter(
                x=self.acc.df.index.to_series() / self.frequency,
                y=self.acc.z,
                mode="lines",
                name="Z Acceleration",
            ),
            row=3,
            col=1,
        )

        max_range = self.acc.max_range()

        self.figure.update_yaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=self.ACCELERATION_UNIT,
            title_text="X",
            range=max_range,
            row=1,
            col=1,
        )
        self.figure.update_yaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=self.ACCELERATION_UNIT,
            title_text="Y",
            range=max_range,
            row=2,
            col=1,
        )
        self.figure.update_yaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=self.ACCELERATION_UNIT,
            title_text="Z",
            range=max_range,
            row=3,
            col=1,
        )
        self.figure.update_xaxes(
            row=3,
            col=1,
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=self.TIME_UNIT,
            title_text="Time",
        )

        self.figure.add_trace(
            Scatter(
                x=self.pos.df.index.to_series() / self.frequency,
                y=self.pos.x,
                mode="lines",
                name="X Distance",
            ),
            row=1,
            col=2,
        )
        self.figure.add_trace(
            Scatter(
                x=self.pos.df.index.to_series() / self.frequency,
                y=self.pos.y,
                mode="lines",
                name="Y Distance",
            ),
            row=2,
            col=2,
        )
        self.figure.add_trace(
            Scatter(
                x=self.pos.df.index.to_series() / self.frequency,
                y=self.pos.z,
                mode="lines",
                name="Z Distance",
            ),
            row=3,
            col=2,
        )

        max_range = self.pos.max_range()

        self.figure.update_yaxes(
            range=max_range,
            row=1,
            col=2,
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="X",
        )
        self.figure.update_yaxes(
            range=max_range,
            row=2,
            col=2,
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="Y",
        )
        self.figure.update_yaxes(
            range=max_range,
            row=3,
            col=2,
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="Z",
        )
        self.figure.update_xaxes(
            row=3,
            col=2,
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=self.TIME_UNIT,
            title_text="Time",
        )

        return self.figure

    def make_2d_animations(self, action: str) -> Figure:

        self.figure = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=["Top view", "Side view", "Front view"],
            specs=[[{"rowspan": 2}, {}], [None, {}]],
        )

        self.figure.layout.title = (
            f"{round(len(self.acc.df) / self.frequency, 1)} Seconds of {action}"
        )

        self.figure.add_trace(
            Scatter(
                x=(0,),
                y=(0,),
                name="XY Position",
                marker=dict(size=Plotter.POINT_SIZE),
            ),
            row=1,
            col=1,
        )
        self.figure.add_trace(
            Scatter(
                x=(0,),
                y=(0,),
                name="YZ Position",
                marker=dict(size=Plotter.POINT_SIZE),
            ),
            row=1,
            col=2,
        )

        self.figure.add_trace(
            Scatter(
                x=(0,),
                y=(0,),
                name="XZ Position",
                marker=dict(size=Plotter.POINT_SIZE),
            ),
            row=2,
            col=2,
        )

        self.figure.frames = [
            Frame(
                data=[
                    Scatter(x=(x,), y=(y,)),
                    Scatter(x=(y,), y=(z,)),
                    Scatter(x=(x,), y=(z,)),
                ]
            )
            for x, y, z in self.pos.triples
        ]

        button = dict(
            label="Play",
            method="animate",
            args=[
                None,
                dict(
                    frame=dict(duration=1000 / self.frequency, redraw=False),
                    transition=dict(duration=0),
                    fromcurrent=True,
                    mode="immediate",
                ),
            ],
        )
        self.figure.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=1.05,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[button],
                )
            ],
        )

        x_range = self.pos.x_range(0.5)
        y_range = self.pos.y_range(0.5)
        z_range = self.pos.z_range(0.03)

        xy_range = max(x_range, y_range, key=VecData.range_magnitude)

        self.figure.update_xaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="X (Front)",
            range=xy_range,
            row=1,
            col=1,
        )
        self.figure.update_yaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="Y (Side)",
            range=xy_range,
            row=1,
            col=1,
        )

        self.figure.update_yaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="Z",
            range=z_range,
            row=1,
            col=2,
        )
        self.figure.update_xaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="Y",
            range=xy_range,
            row=1,
            col=2,
        )

        self.figure.update_xaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="X",
            range=xy_range,
            row=2,
            col=2,
        )
        self.figure.update_yaxes(
            zeroline=True,
            zerolinewidth=3,
            ticksuffix=" m",
            title_text="Z",
            range=z_range,
            row=2,
            col=2,
        )

        return self.figure

    def generate_position_df(
        self, frequency: float, coef: Optional[float] = None
    ) -> VecData:
        def dx(acc: float) -> float:
            c = coef if coef else 1
            return acc * c * (1 / frequency) ** 2

        rename_dict = {col: new for col, new in zip(self.acc.columns, self.POS_COLUMNS)}
        return VecData(
            self.acc.xyz.apply(dx).cumsum().rename(columns=rename_dict),
            self.POS_COLUMNS,
        )
