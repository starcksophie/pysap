#! /usr/bin/env python
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# System import
import os
import re
import sys
import platform
import subprocess
from pprint import pprint
from distutils.version import LooseVersion
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand
from importlib import import_module
try:
    from pip import main as pip_main
except ImportError:
    from pip._internal import main as pip_main


# Package information
release_info = {}
infopath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "pysap", "info.py"))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)
pkgdata = {
    "pysap": [
        os.path.join("test", "*.py"),
        os.path.join("test", "*.json"),
        os.path.join("apps", "*.json")]
}
scripts = [
    os.path.join("pysap", "apps", "pysapview3")
]

# Workaround
if "--release" in sys.argv:
    sys.argv.remove("--release")
    scripts = [
        os.path.join("pysap", "apps", "pysapview3"),
        os.path.join("pysap", "apps", "pysapview")
    ]


class CMakeExtension(Extension):
    """ Use absolute path in setuptools extension.
    """
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """ Define a cmake build extension.
    """

    @staticmethod
    def _preinstall(package_list, options=[]):
        """ Pre-install PyPi packages before running cmake.
        """

        if not isinstance(package_list, list) or not isinstance(options, list):
            raise TypeError('preinstall inputs must be of type list.')

        pip_main(['install'] + options + package_list)

    def _set_pybind_path(self):
        """ Set path to Pybind11 include directory.
        """

        self.pybind_path = getattr(import_module('pybind11'), 'get_include')()

    def run(self):
        """ Redifine the run method.
        """

        # Set preinstall requirements
        preinstall_list = release_info["PREINSTALL_REQUIRES"]

        # Preinstall packages
        self._preinstall(preinstall_list)

        # Set Pybind11 path
        self._set_pybind_path()

        # Check cmake is installed and is sufficiently new.
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)",
                                     out.decode()).group(1))
        if cmake_version < "3.0.0":
            raise RuntimeError("CMake >= 3.0.0 is required.")

        # Build extensions
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """ Build extension with cmake.
        """
        # Define cmake arguments
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                      "-DPYTHON_EXECUTABLE=" + sys.executable,
                      "-DPYBIND11_INCLUDE_DIR=" + self.pybind_path]
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{0}={1}".format(
                cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j8"]

        # Call cmake in specific environment
        env = os.environ.copy()
        env["CXXFLAGS"] = '{0} -DVERSION_INFO=\\"{1}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("Building 'pysparse' in {0}...".format(self.build_temp))
        print("Cmake args:")
        pprint(cmake_args)
        print("Cmake build args:")
        pprint(build_args)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=self.build_temp)
        print()


class HybridTestCommand(TestCommand):
    """ Define custom mix Python/C++ test runner.

    We will execute both Python unittest tests and C++ Catch tests.
    """
    def distutils_dir_name(self, dname):
        """ Returns the name of a distutils build directory.
        """
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(dirname=dname,
                               platform=sysconfig.get_platform(),
                               version=sys.version_info)

    def run(self):
        """ Run hybrid tests.
        """
        # Run Python tests
        super(HybridTestCommand, self).run()
        print("\nPython tests complete, now running C++ tests...\n")

        # Run catch tests
        test_dir = os.path.join("build", self.distutils_dir_name("temp"),
                                "sparse2d", "src", "sparse2d", "tests")
        print("\nExpect C++ test script in {0}.\n".format(test_dir))
        subprocess.call(["./*_test"], cwd=test_dir, shell=True)


# Write setup
setup(
    name=release_info["NAME"],
    description=release_info["DESCRIPTION"],
    long_description=release_info["LONG_DESCRIPTION"],
    license=release_info["LICENSE"],
    classifiers=release_info["CLASSIFIERS"],
    author=release_info["AUTHOR"],
    author_email=release_info["AUTHOR_EMAIL"],
    version=release_info["VERSION"],
    url=release_info["URL"],
    packages=find_packages(exclude="doc"),
    platforms=release_info["PLATFORMS"],
    extras_require=release_info["EXTRA_REQUIRES"],
    install_requires=release_info["REQUIRES"],
    package_data=pkgdata,
    scripts=scripts,
    ext_modules=[CMakeExtension(
        "pysparse", sourcedir=os.path.join("sparse2d", "python"))],
    cmdclass={
        "build_ext": CMakeBuild,
        "test": HybridTestCommand
    }
)
