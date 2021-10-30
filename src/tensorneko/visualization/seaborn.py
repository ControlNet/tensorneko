import seaborn as sns

from .color import Colors

palette = sns.color_palette([
    Colors.BLUE.value,
    Colors.ORANGE.value,
    Colors.GREEN.value,
    Colors.RED.value,
    Colors.PURPLE.value,
    Colors.AZURE.value,
    Colors.GRAY.value,
    Colors.YELLOW.value
])

barplot = sns.barplot
