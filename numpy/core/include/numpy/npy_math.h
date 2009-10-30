#ifndef __NPY_MATH_C99_H_
#define __NPY_MATH_C99_H_

#include <math.h>
#include <numpy/npy_common.h>

/*
 * NAN and INFINITY like macros (same behavior as glibc for NAN, same as C99
 * for INFINITY)
 *
 * XXX: I should test whether INFINITY and NAN are available on the platform
 */
NPY_INLINE static float __npy_inff(void)
{
        const union { npy_uint32 __i; float __f;} __bint = {0x7f800000UL};
        return __bint.__f;
}

NPY_INLINE static float __npy_nanf(void)
{
        const union { npy_uint32 __i; float __f;} __bint = {0x7fc00000UL};
        return __bint.__f;
}

NPY_INLINE static float __npy_pzerof(void)
{
        const union { npy_uint32 __i; float __f;} __bint = {0x00000000UL};
        return __bint.__f;
}

NPY_INLINE static float __npy_nzerof(void)
{
        const union { npy_uint32 __i; float __f;} __bint = {0x80000000UL};
        return __bint.__f;
}

#define NPY_INFINITYF __npy_inff()
#define NPY_NANF __npy_nanf()
#define NPY_PZEROF __npy_pzerof()
#define NPY_NZEROF __npy_nzerof()

#define NPY_INFINITY ((npy_double)NPY_INFINITYF)
#define NPY_NAN ((npy_double)NPY_NANF)
#define NPY_PZERO ((npy_double)NPY_PZEROF)
#define NPY_NZERO ((npy_double)NPY_NZEROF)

#define NPY_INFINITYL ((npy_longdouble)NPY_INFINITYF)
#define NPY_NANL ((npy_longdouble)NPY_NANF)
#define NPY_PZEROL ((npy_longdouble)NPY_PZEROF)
#define NPY_NZEROL ((npy_longdouble)NPY_NZEROF)

/*
 * Useful constants
 */
#define NPY_E         2.718281828459045235360287471352662498  /* e */
#define NPY_LOG2E     1.442695040888963407359924681001892137  /* log_2 e */
#define NPY_LOG10E    0.434294481903251827651128918916605082  /* log_10 e */
#define NPY_LOGE2     0.693147180559945309417232121458176568  /* log_e 2 */
#define NPY_LOGE10    2.302585092994045684017991454684364208  /* log_e 10 */
#define NPY_PI        3.141592653589793238462643383279502884  /* pi */
#define NPY_PI_2      1.570796326794896619231321691639751442  /* pi/2 */
#define NPY_PI_4      0.785398163397448309615660845819875721  /* pi/4 */
#define NPY_1_PI      0.318309886183790671537767526745028724  /* 1/pi */
#define NPY_2_PI      0.636619772367581343075535053490057448  /* 2/pi */
#define NPY_EULER     0.577215664901532860606512090082402431  /* Euler constant */
#define NPY_SQRT2     1.414213562373095048801688724209698079  /* sqrt(2) */
#define NPY_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */

#define NPY_Ef        2.718281828459045235360287471352662498F /* e */
#define NPY_LOG2Ef    1.442695040888963407359924681001892137F /* log_2 e */
#define NPY_LOG10Ef   0.434294481903251827651128918916605082F /* log_10 e */
#define NPY_LOGE2f    0.693147180559945309417232121458176568F /* log_e 2 */
#define NPY_LOGE10f   2.302585092994045684017991454684364208F /* log_e 10 */
#define NPY_PIf       3.141592653589793238462643383279502884F /* pi */
#define NPY_PI_2f     1.570796326794896619231321691639751442F /* pi/2 */
#define NPY_PI_4f     0.785398163397448309615660845819875721F /* pi/4 */
#define NPY_1_PIf     0.318309886183790671537767526745028724F /* 1/pi */
#define NPY_2_PIf     0.636619772367581343075535053490057448F /* 2/pi */
#define NPY_EULERf    0.577215664901532860606512090082402431F /* Euler constan*/
#define NPY_SQRT2f    1.414213562373095048801688724209698079F /* sqrt(2) */
#define NPY_SQRT1_2f  0.707106781186547524400844362104849039F /* 1/sqrt(2) */

#define NPY_El        2.718281828459045235360287471352662498L /* e */
#define NPY_LOG2El    1.442695040888963407359924681001892137L /* log_2 e */
#define NPY_LOG10El   0.434294481903251827651128918916605082L /* log_10 e */
#define NPY_LOGE2l    0.693147180559945309417232121458176568L /* log_e 2 */
#define NPY_LOGE10l   2.302585092994045684017991454684364208L /* log_e 10 */
#define NPY_PIl       3.141592653589793238462643383279502884L /* pi */
#define NPY_PI_2l     1.570796326794896619231321691639751442L /* pi/2 */
#define NPY_PI_4l     0.785398163397448309615660845819875721L /* pi/4 */
#define NPY_1_PIl     0.318309886183790671537767526745028724L /* 1/pi */
#define NPY_2_PIl     0.636619772367581343075535053490057448L /* 2/pi */
#define NPY_EULERl    0.577215664901532860606512090082402431L /* Euler constan*/
#define NPY_SQRT2l    1.414213562373095048801688724209698079L /* sqrt(2) */
#define NPY_SQRT1_2l  0.707106781186547524400844362104849039L /* 1/sqrt(2) */

/*
 * C99 double math funcs
 */
double npy_sin(double x);
double npy_cos(double x);
double npy_tan(double x);
double npy_sinh(double x);
double npy_cosh(double x);
double npy_tanh(double x);

double npy_asin(double x);
double npy_acos(double x);
double npy_atan(double x);
double npy_aexp(double x);
double npy_alog(double x);
double npy_asqrt(double x);
double npy_afabs(double x);

double npy_log(double x);
double npy_log10(double x);
double npy_exp(double x);
double npy_sqrt(double x);

double npy_fabs(double x);
double npy_ceil(double x);
double npy_fmod(double x, double y);
double npy_floor(double x);

double npy_expm1(double x);
double npy_log1p(double x);
double npy_hypot(double x, double y);
double npy_acosh(double x);
double npy_asinh(double xx);
double npy_atanh(double x);
double npy_rint(double x);
double npy_trunc(double x);
double npy_exp2(double x);
double npy_log2(double x);

double npy_atan2(double x, double y);
double npy_pow(double x, double y);
double npy_modf(double x, double* y);

double npy_copysign(double x, double y);

/*
 * IEEE 754 fpu handling. Those are guaranteed to be macros
 */
#ifndef NPY_HAVE_DECL_ISNAN
        #define npy_isnan(x) ((x) != (x))
#else
        #define npy_isnan(x) isnan((x))
#endif

#ifndef NPY_HAVE_DECL_ISFINITE
        #define npy_isfinite(x) !npy_isnan((x) + (-x))
#else
        #define npy_isfinite(x) isfinite((x))
#endif

#ifndef NPY_HAVE_DECL_ISINF
        #define npy_isinf(x) (!npy_isfinite(x) && !npy_isnan(x))
#else
        #define npy_isinf(x) isinf((x))
#endif

#ifndef NPY_HAVE_DECL_SIGNBIT
        int _npy_signbit_f(float x);
        int _npy_signbit_d(double x);
        int _npy_signbit_ld(npy_longdouble x);
        #define npy_signbit(x) \
              (sizeof (x) == sizeof (long double) ? _npy_signbit_ld (x) \
               : sizeof (x) == sizeof (double) ? _npy_signbit_d (x) \
               : _npy_signbit_f (x))
#else
        #define npy_signbit(x) signbit((x))
#endif

/*
 * float C99 math functions
 */

float npy_sinf(float x);
float npy_cosf(float x);
float npy_tanf(float x);
float npy_sinhf(float x);
float npy_coshf(float x);
float npy_tanhf(float x);
float npy_fabsf(float x);
float npy_floorf(float x);
float npy_ceilf(float x);
float npy_rintf(float x);
float npy_truncf(float x);
float npy_sqrtf(float x);
float npy_log10f(float x);
float npy_logf(float x);
float npy_expf(float x);
float npy_expm1f(float x);
float npy_asinf(float x);
float npy_acosf(float x);
float npy_atanf(float x);
float npy_asinhf(float x);
float npy_acoshf(float x);
float npy_atanhf(float x);
float npy_log1pf(float x);
float npy_exp2f(float x);
float npy_log2f(float x);

float npy_atan2f(float x, float y);
float npy_hypotf(float x, float y);
float npy_powf(float x, float y);
float npy_fmodf(float x, float y);

float npy_modff(float x, float* y);

float npy_copysignf(float x, float y);

/*
 * long double C99 math functions
 */

npy_longdouble npy_sinl(npy_longdouble x);
npy_longdouble npy_cosl(npy_longdouble x);
npy_longdouble npy_tanl(npy_longdouble x);
npy_longdouble npy_sinhl(npy_longdouble x);
npy_longdouble npy_coshl(npy_longdouble x);
npy_longdouble npy_tanhl(npy_longdouble x);
npy_longdouble npy_fabsl(npy_longdouble x);
npy_longdouble npy_floorl(npy_longdouble x);
npy_longdouble npy_ceill(npy_longdouble x);
npy_longdouble npy_rintl(npy_longdouble x);
npy_longdouble npy_truncl(npy_longdouble x);
npy_longdouble npy_sqrtl(npy_longdouble x);
npy_longdouble npy_log10l(npy_longdouble x);
npy_longdouble npy_logl(npy_longdouble x);
npy_longdouble npy_expl(npy_longdouble x);
npy_longdouble npy_expm1l(npy_longdouble x);
npy_longdouble npy_asinl(npy_longdouble x);
npy_longdouble npy_acosl(npy_longdouble x);
npy_longdouble npy_atanl(npy_longdouble x);
npy_longdouble npy_asinhl(npy_longdouble x);
npy_longdouble npy_acoshl(npy_longdouble x);
npy_longdouble npy_atanhl(npy_longdouble x);
npy_longdouble npy_log1pl(npy_longdouble x);
npy_longdouble npy_exp2l(npy_longdouble x);
npy_longdouble npy_log2l(npy_longdouble x);

npy_longdouble npy_atan2l(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_hypotl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_powl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_fmodl(npy_longdouble x, npy_longdouble y);

npy_longdouble npy_modfl(npy_longdouble x, npy_longdouble* y);

npy_longdouble npy_copysignl(npy_longdouble x, npy_longdouble y);

/*
 * Non standard functions
 */
double npy_deg2rad(double x);
double npy_rad2deg(double x);
double npy_logaddexp(double x, double y);
double npy_logaddexp2(double x, double y);

float npy_deg2radf(float x);
float npy_rad2degf(float x);
float npy_logaddexpf(float x, float y);
float npy_logaddexp2f(float x, float y);

npy_longdouble npy_deg2radl(npy_longdouble x);
npy_longdouble npy_rad2degl(npy_longdouble x);
npy_longdouble npy_logaddexpl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_logaddexp2l(npy_longdouble x, npy_longdouble y);

#define npy_degrees npy_rad2deg
#define npy_degreesf npy_rad2degf
#define npy_degreesl npy_rad2degl

#define npy_radians npy_deg2rad
#define npy_radiansf npy_deg2radf
#define npy_radiansl npy_deg2radl


/*
 * Complex functions (pointer versions, non-C99)
 */

void npy_csumf_p(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r);
void npy_cdifff_p(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r);
void npy_cnegf_p(npy_cfloat *a, npy_cfloat *r);
void npy_cprodf_p(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r);
void npy_cquotf_p(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r);
void npy_crintf_p(npy_cfloat *x, npy_cfloat *r);
void npy_clog1pf_p(npy_cfloat *x, npy_cfloat *r);
void npy_cexpm1f_p(npy_cfloat *x, npy_cfloat *r);
void npy_cprodif_p(npy_cfloat *x, npy_cfloat *r);
void npy_cexp2f_p(npy_cfloat *x, npy_cfloat *r); /* reserved for future C99 */
void npy_clog10f_p(npy_cfloat *x, npy_cfloat *r); /* reserved for future C99 */
void npy_clog2f_p(npy_cfloat *x, npy_cfloat *r); /* reserved for future C99 */

void npy_csum_p(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r);
void npy_cdiff_p(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r);
void npy_cneg_p(npy_cdouble *a, npy_cdouble *r);
void npy_cprod_p(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r);
void npy_cquot_p(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r);
void npy_crint_p(npy_cdouble *x, npy_cdouble *r);
void npy_clog1p_p(npy_cdouble *x, npy_cdouble *r);
void npy_cexpm1_p(npy_cdouble *x, npy_cdouble *r);
void npy_cprodi_p(npy_cdouble *x, npy_cdouble *r);
void npy_cexp2_p(npy_cdouble *x, npy_cdouble *r); /* reserved for future C99 */
void npy_clog10_p(npy_cdouble *x, npy_cdouble *r); /* reserved for future C99 */
void npy_clog2_p(npy_cdouble *x, npy_cdouble *r); /* reserved for future C99 */

void npy_csuml_p(npy_clongdouble *a, npy_clongdouble *b, npy_clongdouble *r);
void npy_cdiffl_p(npy_clongdouble *a, npy_clongdouble *b, npy_clongdouble *r);
void npy_cnegl_p(npy_clongdouble *a, npy_clongdouble *r);
void npy_cprodl_p(npy_clongdouble *a, npy_clongdouble *b, npy_clongdouble *r);
void npy_cquotl_p(npy_clongdouble *a, npy_clongdouble *b, npy_clongdouble *r);
void npy_crintl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_clog1pl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_cexpm1l_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_cprodil_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_cexp2l_p(npy_clongdouble *x, npy_clongdouble *r); /* reserved for future C99 */
void npy_clog10l_p(npy_clongdouble *x, npy_clongdouble *r); /* reserved for future C99 */
void npy_clog2l_p(npy_clongdouble *x, npy_clongdouble *r); /* reserved for future C99 */


/*
 * Complex functions (pointer versions, C99-like)
 */

void npy_csqrtf_p(npy_cfloat *x, npy_cfloat *r);
void npy_clogf_p(npy_cfloat *x, npy_cfloat *r);
void npy_cexpf_p(npy_cfloat *x, npy_cfloat *r);
void npy_cpowf_p(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r);
void npy_cacosf_p(npy_cfloat *x, npy_cfloat *r);
void npy_cacoshf_p(npy_cfloat *x, npy_cfloat *r);
void npy_casinf_p(npy_cfloat *x, npy_cfloat *r);
void npy_casinhf_p(npy_cfloat *x, npy_cfloat *r);
void npy_catanf_p(npy_cfloat *x, npy_cfloat *r);
void npy_catanhf_p(npy_cfloat *x, npy_cfloat *r);
void npy_ccosf_p(npy_cfloat *x, npy_cfloat *r);
void npy_ccoshf_p(npy_cfloat *x, npy_cfloat *r);
void npy_csinf_p(npy_cfloat *x, npy_cfloat *r);
void npy_csinhf_p(npy_cfloat *x, npy_cfloat *r);
void npy_ctanf_p(npy_cfloat *x, npy_cfloat *r);
void npy_ctanhf_p(npy_cfloat *x, npy_cfloat *r);

void npy_csqrt_p(npy_cdouble *x, npy_cdouble *r);
void npy_clog_p(npy_cdouble *x, npy_cdouble *r);
void npy_cexp_p(npy_cdouble *x, npy_cdouble *r);
void npy_cpow_p(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r);
void npy_cacos_p(npy_cdouble *x, npy_cdouble *r);
void npy_cacosh_p(npy_cdouble *x, npy_cdouble *r);
void npy_casin_p(npy_cdouble *x, npy_cdouble *r);
void npy_casinh_p(npy_cdouble *x, npy_cdouble *r);
void npy_catan_p(npy_cdouble *x, npy_cdouble *r);
void npy_catanh_p(npy_cdouble *x, npy_cdouble *r);
void npy_ccos_p(npy_cdouble *x, npy_cdouble *r);
void npy_ccosh_p(npy_cdouble *x, npy_cdouble *r);
void npy_csin_p(npy_cdouble *x, npy_cdouble *r);
void npy_csinh_p(npy_cdouble *x, npy_cdouble *r);
void npy_ctan_p(npy_cdouble *x, npy_cdouble *r);
void npy_ctanh_p(npy_cdouble *x, npy_cdouble *r);

void npy_csqrtl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_clogl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_cexpl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_cpowl_p(npy_clongdouble *a, npy_clongdouble *b, npy_clongdouble *r);
void npy_cacosl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_cacoshl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_casinl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_casinhl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_catanl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_catanhl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_ccosl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_ccoshl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_csinl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_csinhl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_ctanl_p(npy_clongdouble *x, npy_clongdouble *r);
void npy_ctanhl_p(npy_clongdouble *x, npy_clongdouble *r);

/*
 * Complex functions (non-pointer versions)
 */

#define NPY_COMPLEXFUNC1(type, name) \
    static type name(type a) { type r; name##_p(&a, &r); return r; }
#define NPY_COMPLEXFUNC2(type, name) \
    static type name(type a, type b) { type r; name##_p(&a, &b, &r); return r; }

NPY_COMPLEXFUNC2(npy_cfloat, npy_csumf)
NPY_COMPLEXFUNC2(npy_cfloat, npy_cdifff)
NPY_COMPLEXFUNC1(npy_cfloat, npy_cnegf)
NPY_COMPLEXFUNC2(npy_cfloat, npy_cprodf)
NPY_COMPLEXFUNC2(npy_cfloat, npy_cquotf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_crintf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_clog1pf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_cexpm1f)
NPY_COMPLEXFUNC1(npy_cfloat, npy_cprodif)
NPY_COMPLEXFUNC1(npy_cfloat, npy_cexp2f)
NPY_COMPLEXFUNC1(npy_cfloat, npy_clog10f)
NPY_COMPLEXFUNC1(npy_cfloat, npy_clog2f)

NPY_COMPLEXFUNC2(npy_cdouble, npy_csum)
NPY_COMPLEXFUNC2(npy_cdouble, npy_cdiff)
NPY_COMPLEXFUNC1(npy_cdouble, npy_cneg)
NPY_COMPLEXFUNC2(npy_cdouble, npy_cprod)
NPY_COMPLEXFUNC2(npy_cdouble, npy_cquot)
NPY_COMPLEXFUNC1(npy_cdouble, npy_crint)
NPY_COMPLEXFUNC1(npy_cdouble, npy_clog1p)
NPY_COMPLEXFUNC1(npy_cdouble, npy_cexpm1)
NPY_COMPLEXFUNC1(npy_cdouble, npy_cprodi)
NPY_COMPLEXFUNC1(npy_cdouble, npy_cexp2)
NPY_COMPLEXFUNC1(npy_cdouble, npy_clog10)
NPY_COMPLEXFUNC1(npy_cdouble, npy_clog2)

NPY_COMPLEXFUNC2(npy_clongdouble, npy_csuml)
NPY_COMPLEXFUNC2(npy_clongdouble, npy_cdiffl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_cnegl)
NPY_COMPLEXFUNC2(npy_clongdouble, npy_cprodl)
NPY_COMPLEXFUNC2(npy_clongdouble, npy_cquotl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_crintl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_clog1pl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_cexpm1l)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_cprodil)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_cexp2l)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_clog10l)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_clog2l)

NPY_COMPLEXFUNC1(npy_cfloat, npy_csqrtf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_clogf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_cexpf)
NPY_COMPLEXFUNC2(npy_cfloat, npy_cpowf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_cacosf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_cacoshf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_casinf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_casinhf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_catanf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_catanhf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_ccosf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_ccoshf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_csinf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_csinhf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_ctanf)
NPY_COMPLEXFUNC1(npy_cfloat, npy_ctanhf)

NPY_COMPLEXFUNC1(npy_cdouble, npy_csqrt)
NPY_COMPLEXFUNC1(npy_cdouble, npy_clog)
NPY_COMPLEXFUNC1(npy_cdouble, npy_cexp)
NPY_COMPLEXFUNC2(npy_cdouble, npy_cpow)
NPY_COMPLEXFUNC1(npy_cdouble, npy_cacos)
NPY_COMPLEXFUNC1(npy_cdouble, npy_cacosh)
NPY_COMPLEXFUNC1(npy_cdouble, npy_casin)
NPY_COMPLEXFUNC1(npy_cdouble, npy_casinh)
NPY_COMPLEXFUNC1(npy_cdouble, npy_catan)
NPY_COMPLEXFUNC1(npy_cdouble, npy_catanh)
NPY_COMPLEXFUNC1(npy_cdouble, npy_ccos)
NPY_COMPLEXFUNC1(npy_cdouble, npy_ccosh)
NPY_COMPLEXFUNC1(npy_cdouble, npy_csin)
NPY_COMPLEXFUNC1(npy_cdouble, npy_csinh)
NPY_COMPLEXFUNC1(npy_cdouble, npy_ctan)
NPY_COMPLEXFUNC1(npy_cdouble, npy_ctanh)

NPY_COMPLEXFUNC1(npy_clongdouble, npy_csqrtl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_clogl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_cexpl)
NPY_COMPLEXFUNC2(npy_clongdouble, npy_cpowl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_cacosl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_cacoshl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_casinl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_casinhl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_catanl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_catanhl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_ccosl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_ccoshl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_csinl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_csinhl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_ctanl)
NPY_COMPLEXFUNC1(npy_clongdouble, npy_ctanhl)

#endif
