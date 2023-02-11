// the value is in little endian . 
// So bytes abcd is represented as dcba
//
const FLIP_OTHERS: i32 = 0x00ffffff;
const VALUE_EXTRACTOR: i32 = 0x0000007f;
const SIGN_EXTRACTOR: i32 = 0x00000080;

fn get_left_byte(data: i32) -> i32 {
    let sign = data & SIGN_EXTRACTOR;
    let value = (data & VALUE_EXTRACTOR);
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS << 8u);
    }
}

fn get_mid_left_byte(data: i32) -> i32 {
    let sign = (data & (SIGN_EXTRACTOR << 8u)) >> 8u;
    let value = (data & (VALUE_EXTRACTOR << 8u)) >> 8u;
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS << 8u);
    }
}

fn get_mid_right_byte(data: i32) -> i32 {
    let sign = (data & (SIGN_EXTRACTOR << 16u)) >> 16u;
    let value = (data & (VALUE_EXTRACTOR << 16u)) >> 16u;
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS << 8u);
    }
}

fn get_right_byte(data: i32) -> i32 {
    let sign = (data & (SIGN_EXTRACTOR << 24u)) >> 24u;
    let value = (data & (VALUE_EXTRACTOR << 24u)) >> 24u;
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS << 8u);
    }
}