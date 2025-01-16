// the value is in little endian . 
// So bytes abcd is represented as dcba
//

// TODO use bitcast<i32>(0xffffff00u) when its available in const
const FLIP_OTHERS: i32 = -256; 
const VALUE_EXTRACTOR: i32 = 0x0000007f;
const SIGN_EXTRACTOR: i32 = 0x00000080;
const MAX_I8: i32 = 0x000000ff;

fn get_left_byte(data: i32) -> i32 {
    let sign = data & SIGN_EXTRACTOR;
    let value = (data & VALUE_EXTRACTOR);
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS);
    }
}

fn get_mid_left_byte(data: i32) -> i32 {
    let sign = (data & (SIGN_EXTRACTOR << 8u)) >> 8u;
    let value = (data & (VALUE_EXTRACTOR << 8u)) >> 8u;
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS);
    }
}

fn get_mid_right_byte(data: i32) -> i32 {
    let sign = (data & (SIGN_EXTRACTOR << 16u)) >> 16u;
    let value = (data & (VALUE_EXTRACTOR << 16u)) >> 16u;
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS);
    }
}

fn get_right_byte(data: i32) -> i32 {
    // TODO change let to const when bitcast is available in const
    let sign_extractor = bitcast<i32>(u32(SIGN_EXTRACTOR) << 24u);
    let sign = (data & sign_extractor) >> 24u;
    let value = (data & (VALUE_EXTRACTOR << 24u)) >> 24u;
    if sign == 0 {
        return value;
    } else {
        return value | sign | (FLIP_OTHERS);
    }
}