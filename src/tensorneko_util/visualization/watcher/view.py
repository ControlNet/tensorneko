from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Union, List

from . import Component


@dataclass
class View:
    """
    The view for a server for display.

    Args:
        name (``str``): The name of the view.

    Attributes:
        components (``List[Component]``): The :class:`~tensorneko.visualization.watcher.component.Component` objects in
            the view.

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
        return self._save_json()

    def _save_json(self) -> None:
        if not os.path.exists("watcher"):
            os.mkdir("watcher")

        view_path = os.path.join("watcher", self.name)
        if not os.path.exists(view_path):
            os.mkdir(view_path)

        with open(os.path.join(view_path, "data.json"), "w", encoding="UTF-8") as file:
            file.write(json.dumps({
                "view": self.name,
                "data": list(map(lambda comp: comp.to_dict(), self.components))
            }))

    def add(self, *components: Component) -> View:
        """
        Add the new component to the view.

        Args:
            components (:class:`~tensorneko.visualization.web.component.Component`): The component for adding.

        Returns:
            :class:`~tensorneko.visualization.web.view.View`: The self view object.
        """
        for component in components:
            component.views.append(self)
            self.components.append(component)
            component.update()
            self.update()
        return self

    def add_all(self) -> View:
        """
        Add all defined components.

        Returns:
            :class:`~tensorneko.visualization.web.view.View`: The self view object.
        """
        return self.add(*Component.components.values())

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

    def __getitem__(self, component_name: str) -> Component:
        """
        Get the component by name.

        Args:
            component_name (``str``): The name of the component.

        Returns:
            :class:`~tensorneko.visualization.web.component.Component`: The component object.
        """
        for component in self.components:
            if component.name == component_name:
                return component
        raise KeyError(f"Component {component_name} not found")
