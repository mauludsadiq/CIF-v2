use crate::rdo::types::{LmsTile, FIX_SCALE, TILE_SIZE};

#[inline]
fn lms_to_linear_rgb(l: i64, m: i64, s: i64) -> (f32, f32, f32) {
    let lf = l as f32 / FIX_SCALE as f32;
    let mf = m as f32 / FIX_SCALE as f32;
    let sf = s as f32 / FIX_SCALE as f32;
    let r = ( 5.47221206 * lf - 4.64196010 * mf + 0.16963708 * sf).clamp(0.0, 1.0);
    let g = (-1.12524190 * lf + 2.29317094 * mf - 0.16789520 * sf).clamp(0.0, 1.0);
    let b = ( 0.02980165 * lf - 0.19318073 * mf + 1.16364789 * sf).clamp(0.0, 1.0);
    (r, g, b)
}

#[inline]
fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();
    let ll = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
    let aa = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    let bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
    (ll, aa, bb)
}

#[inline]
fn pixel_d_oklab_fixed(l1: i64, m1: i64, s1: i64, l2: i64, m2: i64, s2: i64) -> i64 {
    let (r1, g1, b1) = lms_to_linear_rgb(l1, m1, s1);
    let (r2, g2, b2) = lms_to_linear_rgb(l2, m2, s2);
    let (ll1, aa1, bb1) = linear_rgb_to_oklab(r1, g1, b1);
    let (ll2, aa2, bb2) = linear_rgb_to_oklab(r2, g2, b2);
    let dl = ll1 - ll2; let da = aa1 - aa2; let db = bb1 - bb2;
    let d2 = dl * dl + da * da + db * db;
    (d2 * FIX_SCALE as f32) as i64
}

pub fn d_oklab_fixed(reference: &LmsTile, decoded: &LmsTile) -> i64 {
    let n = (TILE_SIZE * TILE_SIZE) as i64;
    let mut sum = 0i64;
    for y in 0..TILE_SIZE {
        for x in 0..TILE_SIZE {
            sum = sum.saturating_add(pixel_d_oklab_fixed(
                reference.get(x,y,0), reference.get(x,y,1), reference.get(x,y,2),
                decoded.get(x,y,0),   decoded.get(x,y,1),   decoded.get(x,y,2),
            ));
        }
    }
    sum / n
}

pub fn j_fixed(rate_bits: u64, d_oklab: i64, quality_lambda: i64) -> i64 {
    let rate_term = (rate_bits as i64).saturating_mul(FIX_SCALE);
    let dist_term = quality_lambda.saturating_mul(d_oklab);
    rate_term.saturating_add(dist_term)
}
