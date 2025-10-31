#ifndef MLIR_TENSORRT_SUPPORT_STATUSENUMS_TD
#define MLIR_TENSORRT_SUPPORT_STATUSENUMS_TD

class EnumCase<string sym, int val> {
  string symbol = sym;
  int value = val;
}

class EnumSpec<string sym, string intType_, string docString,
               list<EnumCase> cases_> {
  string intType = intType_;
  string symbol = sym;
  string documentationString = docString;
  list<EnumCase> cases = cases_;
}

def StatusCode : EnumSpec<"StatusCode", "int8_t",
  [{A StatusCode is an enumerated type indicating either no error ("Success") or
    a specific error condition. The purpose of enumerating error conditions is
    to allow functions to communicate to the caller that a potentially recoverable
    error has occured. StatusCode generally should be wrapped in a `Status` object
    along with an optional message payload.
  }],
  [EnumCase<"Success", 0>,
   EnumCase<"InternalError", 1>,
   EnumCase<"InvalidArgument", 2>,
   EnumCase<"Unimplemented", 3>,
   EnumCase<"Unknown", 4>]>;

#endif // MLIR_TENSORRT_SUPPORT_STATUSENUMS_TD
