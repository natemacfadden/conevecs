from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [Extension(
            "latticepts.box_enum",
            sources=["latticepts/box_enum.pyx"],
            include_dirs=["latticepts"],
            define_macros=[("BOX_ENUM_IMPLEMENTATION", None)],
            extra_compile_args=["-O3"],
            language="c",
        )],
        compiler_directives={"language_level": "3"},
    )
)
