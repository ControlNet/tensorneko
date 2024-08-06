import importlib.metadata as metadata

from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import tensorneko_util as N

console = Console()


def _read_requirements(requirements_file: str):
    """Read and parse the requirements from the given file."""
    requirements = [each for each in N.io.read.text(requirements_file).split("\n") if each]
    return [Requirement(req) for req in requirements]


def _get_installed_version(package_name: str):
    """Get the installed version of a package."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None
    except InvalidVersion:
        console.print(f"[bold red]Invalid version for package {package_name}[/bold red]")
        return None


def _check_requirements(requirements):
    """Check for missing and mismatched package versions."""
    missing_packages = []
    version_mismatches = []

    for req in requirements:
        installed_version = _get_installed_version(req.name)
        if installed_version is None:
            missing_packages.append((req.name, req.specifier))
        elif req.specifier and not any(Version(installed_version) in spec for spec in req.specifier):
            version_mismatches.append((req.name, installed_version, req.specifier))

    return missing_packages, version_mismatches


def _display_results(missing_packages, version_mismatches):
    """Display the results using rich."""
    if not missing_packages and not version_mismatches:
        console.print("[bold green]All dependencies are satisfied.[/bold green]")
    else:
        if missing_packages:
            missing_packages_table = Table(show_header=True, header_style="bold red")
            missing_packages_table.add_column("Missing Packages", style="dim", width=40)
            missing_packages_table.add_column("Required Version", style="dim", width=20)
            for pkg, specifier in missing_packages:
                required_versions = str(specifier) if specifier else "Any version"
                missing_packages_table.add_row(pkg, required_versions)
            console.print(
                Panel(missing_packages_table, title=f"[bold red]Missing Packages ({len(missing_packages)})[/bold red]",
                    border_style="red"))

        if version_mismatches:
            version_mismatches_table = Table(show_header=True, header_style="bold yellow")
            version_mismatches_table.add_column("Installed Package", style="dim", width=40)
            version_mismatches_table.add_column("Installed Version", style="dim", width=20)
            version_mismatches_table.add_column("Required Version", style="dim", width=20)
            for pkg, installed_version, specifier in version_mismatches:
                required_versions = str(specifier) if specifier else "Any version"
                version_mismatches_table.add_row(pkg, installed_version, required_versions)
            console.print(Panel(version_mismatches_table,
                title=f"[bold yellow]Version Mismatches ({len(version_mismatches)})[/bold yellow]",
                border_style="yellow"))


def _overwrite_requirements(requirements_file, requirements, mismatched_packages):
    """Overwrite mismatched library versions in the requirements file with the installed versions."""
    new_requirements = []
    mismatched_dict = {pkg: installed_version for pkg, installed_version, _ in mismatched_packages}

    for req in requirements:
        if req.name in mismatched_dict:
            new_requirements.append(f"{req.name}=={mismatched_dict[req.name]}")
        else:
            new_requirements.append(str(req))

    with open(requirements_file, 'w') as file:
        file.write("\n".join(new_requirements))


def dep_check(requirements_file: str, overwrite: bool):
    """Main function to check dependencies against a requirements file."""
    requirements = _read_requirements(requirements_file)

    with console.status("[bold green]Checking dependencies...", spinner="dots"):
        missing_packages, version_mismatches = _check_requirements(requirements)

    _display_results(missing_packages, version_mismatches)

    if overwrite and version_mismatches:
        _overwrite_requirements(requirements_file, requirements, version_mismatches)
        console.print(
            f"[bold green]Requirements file {requirements_file} has been updated with installed versions.[/bold green]")
