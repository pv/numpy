#ifndef NPY_EXTINT128_H_
#define NPY_EXTINT128_H_


typedef struct {
    char sign;
    npy_uint64 lo, hi;
} npy_extint128_t;


/* Integer addition with overflow checking */
static NPY_INLINE npy_int64
safe_add(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a > 0 && b > NPY_MAX_INT64 - a) {
        *overflow_flag = 1;
    }
    else if (a < 0 && b < NPY_MIN_INT64 - a) {
        *overflow_flag = 1;
    }
    return a + b;
}


/* Integer subtraction with overflow checking */
static NPY_INLINE npy_int64
safe_sub(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a >= 0 && b < a - NPY_MAX_INT64) {
        *overflow_flag = 1;
    }
    else if (a < 0 && b > a - NPY_MIN_INT64) {
        *overflow_flag = 1;
    }
    return a - b;
}


/* Integer multiplication with overflow checking */
static NPY_INLINE npy_int64
safe_mul(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a > 0) {
        if (b > NPY_MAX_INT64 / a || b < NPY_MIN_INT64 / a) {
            *overflow_flag = 1;
        }
    }
    else if (a < 0) {
        if (b > 0 && a < NPY_MIN_INT64 / b) {
            *overflow_flag = 1;
        }
        else if (b < 0 && a < NPY_MAX_INT64 / b) {
            *overflow_flag = 1;
        }
    }
    return a * b;
}


/* Long integer init */
static NPY_INLINE npy_extint128_t
to_128(npy_int64 x)
{
    npy_extint128_t result;
    result.sign = (x >= 0 ? 1 : -1);
    if (x >= 0) {
        result.lo = x;
    }
    else {
        result.lo = (npy_uint64)(-(x + 1)) + 1;
    }
    result.hi = 0;
    return result;
}


/* Long integer init */
static NPY_INLINE npy_extint128_t
to_u128(npy_uint64 x)
{
    npy_extint128_t result;
    result.sign = 1;
    result.hi = 0;
    result.lo = x;
    return result;
}


static NPY_INLINE npy_int64
to_64(npy_extint128_t x, char *overflow)
{
    if (x.hi != 0 ||
        (x.sign > 0 && x.lo > NPY_MAX_INT64) ||
        (x.sign < 0 && x.lo != 0 && x.lo - 1 > -(NPY_MIN_INT64 + 1))) {
        *overflow = 1;
    }
    return x.lo * x.sign;
}




/* Long integer multiply */
static NPY_INLINE npy_extint128_t
mul_u64_u64(npy_uint64 a, npy_uint64 b)
{
    npy_extint128_t z;
    npy_uint64 x1, x2, y1, y2, r1, r2, prev;

    x1 = a & 0xffffffff;
    x2 = a >> 32;

    y1 = b & 0xffffffff;
    y2 = b >> 32;

    r1 = x1*y2;
    r2 = x2*y1;

    z.sign = 1;
    z.hi = x2*y2 + (r1 >> 32) + (r2 >> 32);
    z.lo = x1*y1;

    /* Add with carry */
    prev = z.lo;
    z.lo += (r1 << 32);
    if (z.lo < prev) {
        ++z.hi;
    }

    prev = z.lo;
    z.lo += (r2 << 32);
    if (z.lo < prev) {
        ++z.hi;
    }

    return z;
}


/* Long integer multiply */
static NPY_INLINE npy_extint128_t
mul_64_64(npy_int64 a, npy_int64 b)
{
    npy_extint128_t x, y, z;

    x = to_128(a);
    y = to_128(b);

    z = mul_u64_u64(x.lo, y.lo);
    z.sign = x.sign * y.sign;

    return z;
}


/* Long integer add */
static NPY_INLINE npy_extint128_t
add_128(npy_extint128_t x, npy_extint128_t y, char *overflow)
{
    npy_extint128_t z;

    if (x.sign == y.sign) {
        z.sign = x.sign;
        z.hi = x.hi + y.hi;
        if (z.hi < x.hi) {
            *overflow = 1;
        }
        z.lo = x.lo + y.lo;
        if (z.lo < x.lo) {
            if (z.hi == NPY_MAX_UINT64) {
                *overflow = 1;
            }
            ++z.hi;
        }
    }
    else if (x.hi > y.hi || (x.hi == y.hi && x.lo >= y.lo)) {
        z.sign = x.sign;
        z.hi = x.hi - y.hi;
        z.lo = x.lo;
        z.lo -= y.lo;
        if (z.lo > x.lo) {
            --z.hi;
        }
    }
    else {
        z.sign = y.sign;
        z.hi = y.hi - x.hi;
        z.lo = y.lo;
        z.lo -= x.lo;
        if (z.lo > y.lo) {
            --z.hi;
        }
    }

    return z;
}


/* Long integer negation */
static NPY_INLINE npy_extint128_t
neg_128(npy_extint128_t x)
{
    npy_extint128_t z = x;
    z.sign *= -1;
    return z;
}


static NPY_INLINE npy_extint128_t
sub_128(npy_extint128_t x, npy_extint128_t y, char *overflow)
{
    return add_128(x, neg_128(y), overflow);
}


static NPY_INLINE npy_extint128_t
shl_128(npy_extint128_t v, unsigned int n)
{
    npy_extint128_t z;
    z = v;
    if (n == 0) {
        /* noop */
    }
    else if (n < 64) {
        z.hi <<= n;
        z.hi |= z.lo >> (64 - n);
        z.lo <<= n;
    }
    else {
        z.hi = z.lo << (n - 64);
        z.lo = 0;
    }
    return z;
}


static NPY_INLINE npy_extint128_t
shr_128(npy_extint128_t v, unsigned int n)
{
    npy_extint128_t z;
    z = v;
    if (n == 0) {
        /* noop */
    }
    else if (n < 64) {
        z.lo >>= n;
        z.lo |= z.hi << (64 - n);
        z.hi >>= n;
    }
    else {
        z.lo = z.hi >> (n - 64);
        z.hi = 0;
    }
    return z;
}

static NPY_INLINE int
gt_128(npy_extint128_t a, npy_extint128_t b)
{
    if (a.sign > 0 && b.sign > 0) {
        return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
    }
    else if (a.sign < 0 && b.sign < 0) {
        return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
    }
    else if (a.sign > 0 && b.sign < 0) {
        return a.hi != 0 || a.lo != 0 || b.hi != 0 || b.lo != 0;
    }
    else {
        return 0;
    }
}


/* Long integer divide */
static NPY_INLINE npy_extint128_t
divmod_128_64(npy_extint128_t x, npy_int64 b_in, npy_int64 *mod)
{
    npy_extint128_t remainder, pointer, result, divisor, tmp;
    npy_uint64 b, q, r, v;
    npy_uint32 b0, b1, vnext;
    int j, shift;
    char overflow = 0;

    assert(b_in > 0);

    b = b_in;

    if (b <= 1 || x.hi == 0) {
        result.sign = x.sign;
        result.lo = x.lo / b;
        result.hi = x.hi / b;
        *mod = x.sign * (x.lo % b);
        return result;
    }

    /* Pre-division of the high bits */

    q = x.hi / b;
    r = x.hi % b;

    result.sign = 1;
    result.hi = q;
    result.lo = 0;

    b1 = b >> 32;
    b0 = b & 0xffffffff;

    if (b1 == 0) {
        /* Long division in base 32 with 1-digit divisor */

        r = (r << 32) | (x.lo >> 32);

        q = r / b0;
        r = r % b0;

        result.lo = q << 32;

        r = (r << 32) | (x.lo & 0xffffffff);

        q = r / b0;
        r = r % b0;

        result.lo |= q;

        /* Fix signs and return */
        result.sign = x.sign;
        *mod = x.sign * r;
        return result;
    }

    remainder.sign = 1;
    remainder.hi = r;
    remainder.lo = x.lo;

    /* Long division in base 32, with 2-digit divisor, loop unrolled.

       See Knuth, TAOCP vol 2 sec 4.3.1 Algorithm D.
       We don't need step D5-D6 since the divisor is only two digits wide.
     */

    /* Note that at this point we have remainder.hi < b <= 2**63-1 */

    /* Normalize so that b1 >= 2**31 */
    shift = 0;
    while ((b1 & ((npy_uint32)1 << 31)) == 0) {
        b1 <<= 1;
        ++shift;
    }
    if (shift != 0) {
        b1 |= b0 >> (32 - shift);
        b0 <<= shift;
        b <<= shift;
        remainder = shl_128(remainder, shift);
    }

    /* Division of the highest three digits */
    v = remainder.hi;
    q = v / b1;  /* < (2**63-1) // 2**31 < 2**32 */
    r = v % b1;

    assert((q >> 32) == 0);

    vnext = (remainder.lo >> 32);

    while (q*b0 > ((r << 32) | vnext)) {
        /* These loops require at most 2 iterations, cf TAOCP */
        --q;
        r += b1;
        if ((r >> 32) != 0) {
            break;
        }
    }

    /* Subtract q*b. This is guaranteed to remove the high bits of the
       remainder, so we just drop them here. */
    v = ((r << 32) | vnext) - q*b0;
    remainder.hi = v >> 32;
    remainder.lo = (v << 32) | (remainder.lo & 0xffffffff);

    result.lo |= q << 32;

    /* Division of the lowest three digits */
    v = (remainder.hi << 32) | (remainder.lo >> 32);
    q = v / b1;
    r = v % b1;

    vnext = (remainder.lo & 0xffffffff);

    while ((q >> 32) != 0 || q*b0 > ((r << 32) | vnext)) {
        --q;
        r += b1;
        if ((r >> 32) != 0) {
            break;
        }
    }

    remainder.hi = 0;
    remainder.lo = ((r << 32) | vnext) - q*b0;

    result.lo |= q;

    /* Because b is larger than 32-bit, the rest is remainder */

    remainder.lo >>= shift; /* Unnormalize */

    /* Fix signs and return; cannot overflow */
    result.sign = x.sign;
    *mod = x.sign * remainder.lo;

    return result;
}


/* Divide and round down (positive divisor; no overflows) */
static NPY_INLINE npy_extint128_t
floordiv_128_64(npy_extint128_t a, npy_int64 b)
{
    npy_extint128_t result;
    npy_int64 remainder;
    char overflow = 0;
    assert(b > 0);
    result = divmod_128_64(a, b, &remainder);
    if (a.sign < 0 && remainder != 0) {
        result = sub_128(result, to_128(1), &overflow);
    }
    return result;
}


/* Divide and round up (positive divisor; no overflows) */
static NPY_INLINE npy_extint128_t
ceildiv_128_64(npy_extint128_t a, npy_int64 b)
{
    npy_extint128_t result;
    npy_int64 remainder;
    char overflow = 0;
    assert(b > 0);
    result = divmod_128_64(a, b, &remainder);
    if (a.sign > 0 && remainder != 0) {
        result = add_128(result, to_128(1), &overflow);
    }
    return result;
}

#endif
