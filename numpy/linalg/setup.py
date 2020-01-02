import os
import sys

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, system_info
    config = Configuration('linalg', parent_package, top_path)

    config.add_data_dir('tests')

    # Configure lapack_lite

    src_dir = 'lapack_lite'
    lapack_lite_src = [
        os.path.join(src_dir, 'python_xerbla.c'),
        os.path.join(src_dir, 'f2c_z_lapack.c'),
        os.path.join(src_dir, 'f2c_c_lapack.c'),
        os.path.join(src_dir, 'f2c_d_lapack.c'),
        os.path.join(src_dir, 'f2c_s_lapack.c'),
        os.path.join(src_dir, 'f2c_lapack.c'),
        os.path.join(src_dir, 'f2c_blas.c'),
        os.path.join(src_dir, 'f2c_config.c'),
        os.path.join(src_dir, 'f2c.c'),
    ]
    all_sources = config.paths(lapack_lite_src)

    if os.environ.get('NPY_USE_BLAS_ILP64', "0") != "0":
        lapack_info = get_info('lapack_ilp64_opt', 2)
    else:
        lapack_info = get_info('lapack_opt', 0)  # and {}

    use_lapack_lite = not lapack_info

    if use_lapack_lite:
        # This makes numpy.distutils write the fact that lapack_lite
        # is being used to numpy.__config__
        class numpy_linalg_lapack_lite(system_info):
            def calc_info(self):
                info = {'language': 'c'}
                if sys.maxsize > 2**32:
                    # Build lapack-lite in 64-bit integer mode
                    info['define_macros'] = [('HAVE_BLAS_ILP64', None)]
                self.set_info(**info)

        lapack_info = numpy_linalg_lapack_lite().get_info(2)

    def get_lapack_lite_sources(ext, build_dir):
        set_fortran_character_len_type(ext)

        if use_lapack_lite:
            print("### Warning:  Using unoptimized lapack ###")
            return all_sources
        else:
            if sys.platform == 'win32':
                print("### Warning:  python_xerbla.c is disabled ###")
                return []
            return [all_sources[0]]

    def set_fortran_character_len_type(ext):
        """
        Set the type of the hidden character string length parameter in
        Fortran ABI in calls to LAPACK. Generally, for the use case in
        the LAPACK calls here, the argument has historically not
        mattered as it is not referenced by most compilers, but this is
        not guaranteed by any standard.

        See gh-13809,
        https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90329,
        https://developer.r-project.org/Blog/public/2019/09/25/gfortran-issues-with-lapack-ii/
        for details. Here, we follow the behavior of f2py and
        assume size_t is the right type. However, we will try to check
        at build time that it appears to be so.
        """
        # Check Fortran hidden character parameter information
        config_cmd = config.get_config_cmd()

        if use_lapack_lite:
            ok = True
            c_type = "size_t"
        elif 'NPY_FORTRAN_CHARACTER_LEN_TYPE' in os.environ:
            ok = True
            c_type = os.environ['NPY_FORTRAN_CHARACTER_LEN_TYPE']
        else:
            c_type = "size_t"
            ok = config_cmd.check_fortran_character_hidden_parameter_type(c_type)

        if ok is None:
            # Could not find out, fall back to size_t
            print("## Warning: could not determine Fortran compiler "
                  "character length argument type, assuming size_t")
            ok = True
            c_type = "size_t"

        if ok:
            if c_type:
                ext.define_macros += [('BLAS_CHARACTER_LEN_TYPE', c_type)]
        else:
            raise RuntimeError(
                "Fortran compiler has incompatible  character length calling "
                "convention. Use environment variable NPY_FORTRAN_CHARACTER_LEN_TYPE "
                "to set it manually."
            )

    # lapack_lite
    config.add_extension(
        'lapack_lite',
        sources=['lapack_litemodule.c', get_lapack_lite_sources],
        depends=['lapack_lite/f2c.h'],
        extra_info=lapack_info,
    )

    # umath_linalg module
    config.add_extension(
        '_umath_linalg',
        sources=['umath_linalg.c.src', get_lapack_lite_sources],
        depends=['lapack_lite/f2c.h'],
        extra_info=lapack_info,
        libraries=['npymath'],
    )
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
