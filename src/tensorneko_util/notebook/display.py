import IPython
from IPython.display import Audio, Video, YouTubeVideo, Code, Image, HTML
import sys


def audio(path: str):
    return IPython.display.display(Audio(path))


def video(path: str, embed=False):
    return IPython.display.display(Video(path, embed=embed))


def youtube_video(id_: str):
    return IPython.display.display(YouTubeVideo(id_))


def image(path: str):
    return IPython.display.display(Image(path))


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

def html(source: str):
    # render HTML in Jupyter
    return IPython.display.HTML(source)


class Display:
    audio = staticmethod(audio)
    video = staticmethod(video)
    image = staticmethod(image)
    youtube_video = staticmethod(youtube_video)
    code = staticmethod(code)
    html = staticmethod(html)

    def __call__(self, path: str, *args, **kwargs):
        ext = path.split(".")[-1]
        if ext in ("mp3", "wav", "ogg"):
            return self.audio(path)
        elif ext in ("mp4", "avi", "mkv"):
            return self.video(path, *args, **kwargs)
        elif ext in ("jpg", "jpeg", "png", "bmp", "gif"):
            return self.image(path)
        elif ext in ("py", "c", "cpp", "java", "js", "html", "css", "json", "yaml", "yml", "xml", "sh", "bat"):
            return self.code(path, *args, **kwargs)
        else:
            raise ValueError(f"Unknown file type: {ext}")


sys.modules[__name__] = Display()
