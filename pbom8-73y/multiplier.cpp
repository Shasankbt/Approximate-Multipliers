#include <torch/extension.h>


// ==========================================
// PBOM8 approximate mantissa multiply
// ==========================================
inline uint8_t pbom8_mantissa_mult(uint8_t a, uint8_t b) {
    const uint8_t b_low  = b & 0x0F;
    const uint8_t b_high = b >> 4;

    const uint16_t m1_exact  = (a * b_low) & 0xFFF8;
    const uint16_t m1_approx = (a & 0x07) | (b_low & 0x07);
    const uint16_t m1_out    = m1_exact | m1_approx;

    const uint16_t m2_exact  = (a * b_high) & 0xFF00;
    const uint16_t m2_approx = a | b_high;
    const uint16_t m2_out    = m2_exact | m2_approx;

    return static_cast<uint8_t>((m1_out | (m2_out << 4)) >> 8);
}

// ==========================================
// Wrapper for conversion of fp16 to bits
// ==========================================
union HalfBits {
    at::Half h;
    uint16_t u;
};

/*
    | Sign (1 bit) | Exponent (5 bits) | Mantissa/Fraction (10 bits) |
    |    bit 15    |    bits 14-10     |        bits 9-0             |

    and the value is represented as 
    (-1)^sign × 2^(exponent - 15) × (1.mantissa)
    
    Exponent bias = 15 (for FP16)
    ea and eb are the raw 5-bit exponent values (0-31)
    Actual exponent = ea - 15 (ranges from -15 to +16)
*/
float approx_half_scalar(at::Half ha, at::Half hb) {
    HalfBits a{ha}, b{hb}, out;

    const uint16_t sign = ((a.u ^ b.u) >> 15) & 0x1;
    const uint16_t ea = (a.u >> 10) & 0x1F;
    const uint16_t eb = (b.u >> 10) & 0x1F;

    const uint16_t ma = 0x0400 | (a.u & 0x03FF);
    const uint16_t mb = 0x0400 | (b.u & 0x03FF);

    // ================================
    /* ignore three least significant bits */
    // const uint8_t ma8 = ma >> 3;
    // const uint8_t mb8 = mb >> 3;
    // --------------------------------
    /* ignore three most significant bits */
    const uint8_t ma8 = ma & 0xFF; 
    const uint8_t mb8 = mb & 0xFF;
    // ================================


    uint8_t prod8 = pbom8_mantissa_mult(ma8, mb8);

    // exp = (ea - 15) + (eb - 15) + 15 => ea + eb - 15
    int exp = int(ea) + int(eb) - 15;

    if (!(prod8 & 0x80)) {
        prod8 <<= 1;
        exp -= 1;
    }

    if (exp <= 0) return 0.0f;
    if (exp >= 31) exp = 31, prod8 = 0;

    const uint16_t frac = (uint16_t(prod8 & 0x7F)) << 3;

    out.u = (sign << 15) | (uint16_t(exp) << 10) | (frac & 0x03FF);
    return static_cast<float>(out.h);
}