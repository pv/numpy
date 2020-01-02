"""This module implements additional tests ala autoconf which can be useful.

"""
import textwrap
import subprocess
import distutils.ccompiler

# We put them here since they could be easily reused outside numpy.distutils

def check_inline(cmd):
    """Return the inline identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #ifndef __cplusplus
        static %(inline)s int static_func (void)
        {
            return 0;
        }
        %(inline)s int nostatic_func (void)
        {
            return 0;
        }
        #endif""")

    for kw in ['inline', '__inline__', '__inline']:
        st = cmd.try_compile(body % {'inline': kw}, None, None)
        if st:
            return kw

    return ''


def check_restrict(cmd):
    """Return the restrict identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        static int static_func (char * %(restrict)s a)
        {
            return 0;
        }
        """)

    for kw in ['restrict', '__restrict__', '__restrict']:
        st = cmd.try_compile(body % {'restrict': kw}, None, None)
        if st:
            return kw

    return ''


def check_compiler_gcc4(cmd):
    """Return True if the C compiler is GCC 4.x."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        int
        main()
        {
        #if (! defined __GNUC__) || (__GNUC__ < 4)
        #error gcc >= 4 required
        #endif
            return 0;
        }
        """)
    return cmd.try_compile(body, None, None)


def check_gcc_function_attribute(cmd, attribute, name):
    """Return True if the given function attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s %s(void*);

        int
        main()
        {
            return 0;
        }
        """) % (attribute, name)
    return cmd.try_compile(body, None, None) != 0


def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code,
                                                include):
    """Return True if the given function attribute is supported with
    intrinsics."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #include<%s>
        int %s %s(void)
        {
            %s;
            return 0;
        }

        int
        main()
        {
            return 0;
        }
        """) % (include, attribute, name, code)
    return cmd.try_compile(body, None, None) != 0


def check_gcc_variable_attribute(cmd, attribute):
    """Return True if the given variable attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s foo;

        int
        main()
        {
            return 0;
        }
        """) % (attribute, )
    return cmd.try_compile(body, None, None) != 0


def check_fortran_character_hidden_parameter_type(cmd, c_type):
    """
    Check the data type of the hidden Fortran character length parameter.

    Parameters
    ----------
    cmd : numpy.distutils.commands.config.config
        Config command.
    c_type : str
        Name of the C type to try (e.g. size_t).

    Returns
    -------
    validity : {None, True, False}
        Returns whether the Fortran compiler character string length parameter
        data type matched (True/False), or whether it was not possible to
        check (None) due to missing compilers.

    """
    cmd._check_compiler()

    c_compiler = cmd.compiler
    f_compiler = cmd.fcompiler

    if c_compiler is None:
        return None

    if f_compiler is None:
        return None

    configtest_f = textwrap.dedent(
        """
        c Test file
              subroutine tstf()
              external tstc
              call tstc('a', 'bb')
              end
        """
    )

    configtest_c = textwrap.dedent(
        """
        #include <stdlib.h>
        #include <string.h>

        #ifdef NO_APPEND_FORTRAN
        #define FNAME(name) name
        #else
        #define FNAME(name) name ## _
        #endif

        void FNAME(tstf)();

        int FNAME(tstc)(char *s1, char *s2, %(c_type)s len1, %(c_type)s len2)
        {
            if (len1 != 1) exit(1);
            if (len2 != 2) exit(1);
            if (strncmp(s1, "a", len1) != 0) exit(1);
            if (strncmp(s2, "bb", len2) != 0) exit(1);
            return 0;
        }

        int main() {
            FNAME(tstf)();
            return 0;
        }
        """
        % dict(c_type=c_type)
    )

    configtest_f_fn = "_configtest_f.f"
    configtest_c_fn = "_configtest_c.c"
    prog_basename = "_configtest"
    prog = prog_basename + (c_compiler.exe_extension or "")

    cmd.temp_files.extend([configtest_c, configtest_f, prog])

    with open(configtest_f_fn, "wt") as f:
        f.write(configtest_f)

    with open(configtest_c_fn, "wt") as f:
        f.write(configtest_c)

    obj_c = c_compiler.compile([configtest_c_fn])
    cmd.temp_files.extend(obj_c)
    obj_f = f_compiler.compile([configtest_f_fn])
    cmd.temp_files.extend(obj_f)

    try:
        c_compiler.link_executable(obj_c + obj_f, prog_basename)
    except distutils.ccompiler.LinkError:
        # Failed to link --- possibly incompatible Fortran compiler
        return None

    cmd.temp_files.append(prog)

    res = subprocess.call([prog])
    return res == 0
