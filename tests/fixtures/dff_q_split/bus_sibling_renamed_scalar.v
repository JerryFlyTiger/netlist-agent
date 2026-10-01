module top(a, b, c, clk, y);
  input a, b, c, clk;
  output y;
  wire [1:0] w;
  wire t, s;
  and g1(s, a, b);
  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(s), .Q(w[1]));
  xor g2(t, s, w[1]);
  or g3(y, t, c);
endmodule
