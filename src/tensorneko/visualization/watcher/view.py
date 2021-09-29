from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union, List
import os

from ...io import write
if TYPE_CHECKING:
    from . import Component


@dataclass
class View:
    """
    The view for a server for display.

    Args:
        name (``str``): The name of the view.

    Attributes:
        components (``List[Component]``): The :class:`~tensorneko.visualization.watcher.component.Component` objects in the
            view.

    Examples::

        # create components
        var = tensorneko.visualization.watcher.Variable("x", 5)
        pb = tensorneko.visualization.watcher.ProgressBar("process", 0, 10000)

        # create a view and add components
        view = tensorneko.visualization.watcher.View("view1")
        view.add(var).add(pb)

        # components also support read by multiple views
        view2 = tensorneko.visualization.watcher.View("view2")
        view2.add(var)

    """
    name: str
    components: List[Component] = field(init=False, default_factory=list)

    def update(self) -> None:
        """Save the state to the json file."""
        return self._to_json()

    def _to_json(self) -> None:
        if not os.path.exists(self.name):
            os.mkdir(self.name)

        write.text.to_json(os.path.join(self.name, "data.json"),
            list(map(lambda comp: comp.to_dict(), self.components))
        )

    def add(self, component: Component) -> View:
        """
        Add the new component to the view.

        Args:
            component (:class:`~tensorneko.visualization.web.component.Component`): The component for adding.

        Returns:
            :class:`~tensorneko.visualization.web.view.View`: The self view object.
        """
        component.views.append(self)
        self.components.append(component)
        component.update()
        self.update()
        return self

    def remove(self, component: Union[Component, str]) -> View:
        """
        Remove the component in the view.

        Args:
            component (:class:`~tensorneko.visualization.web.component.Component` | ``str``):
                The component for removing.

        Returns:
            :class:`~tensorneko.visualization.web.view.View`: The self view object.
        """
        if type(component) is Component and component in self.components:
            self.components.remove(component)
        elif type(component) is str:
            for i in range(len(self.components)):
                if self.components[i].name == component:
                    self.components[i].views.remove(self)
                    del self.components[i]
                    break
            else:
                print("Component remove failed")
        else:
            print("Component remove failed")

        self.update()
        return self
