//! Basic numeric-type sanity checks, mirroring fplll's test_nr.cpp.

use fplll::integer::Z;

#[test]
fn z_basic_ops() {
    let a = Z::from_i64(42);
    let b = Z::from_i64(7);
    let mut c = Z::zero();
    c.mul_set(&a, &b);
    assert_eq!(c.to_str_radix(10), "294");
    c.add_set(&a, &b);
    assert_eq!(c.to_str_radix(10), "49");
    c.sub_set(&a, &b);
    assert_eq!(c.to_str_radix(10), "35");
}

#[test]
fn z_addmul_submul() {
    let mut acc = Z::from_i64(100);
    let x = Z::from_i64(3);
    let y = Z::from_i64(5);
    acc.addmul(&x, &y);
    assert_eq!(acc.to_str_radix(10), "115");
    acc.submul(&x, &y);
    assert_eq!(acc.to_str_radix(10), "100");
    acc.addmul_si(&x, 7);
    assert_eq!(acc.to_str_radix(10), "121");
    acc.submul_si(&x, 7);
    assert_eq!(acc.to_str_radix(10), "100");
}

#[test]
fn z_mul_2si() {
    let x = Z::from_i64(3);
    let mut r = Z::zero();
    r.mul_2si(&x, 10);
    assert_eq!(r.to_str_radix(10), "3072");
    r.mul_2si(&x, -1);
    assert_eq!(r.to_str_radix(10), "1");
}

#[test]
fn z_get_d_2exp() {
    let x = Z::from_i64(1 << 30);
    let (m, e) = x.get_d_2exp();
    let val = m * 2f64.powi(e as i32);
    assert!((val - (1i64 << 30) as f64).abs() < 1.0);
}

#[test]
fn z_parse_big() {
    let s = "123456789012345678901234567890";
    let z = Z::from_str(s).unwrap();
    assert_eq!(z.to_str_radix(10), s);
}

#[test]
fn z_next_prime() {
    let x = Z::from_i64(10);
    let p = x.next_prime();
    assert_eq!(p.to_str_radix(10), "11");
}
