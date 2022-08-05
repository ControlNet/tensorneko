import IPython
from IPython.display import Audio, Video, YouTubeVideo, Code


def audio(path: str):
    return IPython.display.display(Audio(path))


def video(path: str, embed=False):
    return IPython.display.display(Video(path, embed=embed))


def youtube_video(id_: str):
    return IPython.display.display(YouTubeVideo(id_))


def code(path: str, language: str = None):
    from pygments import highlight
    from pygments.formatters.html import HtmlFormatter
    # fix JupyterLab from https://github.com/ipython/ipython/issues/11747
    def _jupyterlab_repr_html_(self):
        fmt = HtmlFormatter()
        style = "<style>{}\n{}</style>".format(
            fmt.get_style_defs(".output_html"), fmt.get_style_defs(".jp-RenderedHTML")
        )
        return style + highlight(self.data, self._get_lexer(), fmt)

    IPython.display.Code._repr_html_ = _jupyterlab_repr_html_
    return IPython.display.display(Code(path, language=language))
