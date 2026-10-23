//! CPython mode-0 shortest decimal's nearest/even choice at an exact decimal
//! midpoint. Rust Debug may select the odd decimal instead. Keep its shortest
//! round-trip digits except at this provable tie, and require that the even
//! alternative also round-trips (power-of-two intervals are asymmetric).
//! There is no floating-point tolerance or decimal-to-integer truncation here.
pub(crate) fn shortest(value: f64) -> String {
    let original = format!("{value:?}");
    if !value.is_finite() || value == 0.0 {
        return original;
    }
    let raw = original.strip_prefix('-').unwrap_or(&original);
    let (mantissa, exponent) = match raw.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap()),
        None => (raw, 0),
    };
    let fraction = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len() as i32);
    let digits = mantissa.replace('.', "");
    let number = match digits.parse::<u64>() {
        Ok(n) if n % 2 == 1 => n,
        _ => return original,
    };
    let power = exponent - fraction;
    let bits = value.to_bits();
    let field = ((bits >> 52) & 2047) as i32;
    let significand = (bits & ((1_u64 << 52) - 1)) | if field == 0 { 0 } else { 1_u64 << 52 };
    let binary_power = if field == 0 { -1074 } else { field - 1023 - 52 };
    let trailing = significand.trailing_zeros();
    if power != binary_power + 1 + trailing as i32 {
        return original;
    }
    let odd = u128::from(significand >> trailing);
    let Some(five) = 5_u128.checked_pow(power.unsigned_abs()) else {
        return original;
    };
    let double = u128::from(number) * 2;
    let tie = if power < 0 {
        match odd.checked_mul(five) {
            Some(n) if n == double - 1 => Some(number - 1),
            Some(n) if n == double + 1 => Some(number + 1),
            _ => None,
        }
    } else if (double - 1).checked_mul(five) == Some(odd) {
        Some(number - 1)
    } else if (double + 1).checked_mul(five) == Some(odd) {
        Some(number + 1)
    } else {
        None
    };
    let Some(even) = tie else {
        return original;
    };
    let candidate = render(even, power, value.is_sign_negative());
    if candidate.parse::<f64>().is_ok_and(|n| n.to_bits() == bits) {
        candidate
    } else {
        original
    }
}
fn render(number: u64, power: i32, negative: bool) -> String {
    let mut digits = number.to_string();
    let point = digits.len() as i32 + power;
    while digits.ends_with('0') && digits.len() > 1 {
        digits.pop();
    }
    let mut output = if point <= -4 || point > 16 {
        let mut mantissa = digits[..1].to_string();
        if digits.len() > 1 {
            mantissa.push('.');
            mantissa.push_str(&digits[1..]);
        }
        format!("{mantissa}e{:+03}", point - 1)
    } else if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    };
    if negative {
        output.insert(0, '-');
    }
    output
}
