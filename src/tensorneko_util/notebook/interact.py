import IPython
from IPython.display import Audio, Video, YouTubeVideo, Code
from pygments import highlight
from pygments.formatters.html import HtmlFormatter


class Display:

    @staticmethod
    def audio(path: str):
        return IPython.display.display(Audio(path))

    @staticmethod
    def video(path: str, embed=False):
        return IPython.display.display(Video(path, embed=embed))

    @staticmethod
    def youtube_video(id_: str):
        return IPython.display.display(YouTubeVideo(id_))

    @staticmethod
    def code(path: str, language: str = None):
        # fix JupyterLab from https://github.com/ipython/ipython/issues/11747
        def _jupyterlab_repr_html_(self):
            fmt = HtmlFormatter()
            style = "<style>{}\n{}</style>".format(
                fmt.get_style_defs(".output_html"), fmt.get_style_defs(".jp-RenderedHTML")
            )
            return style + highlight(self.data, self._get_lexer(), fmt)
        IPython.display.Code._repr_html_ = _jupyterlab_repr_html_
        return IPython.display.display(Code(path, language=language))
