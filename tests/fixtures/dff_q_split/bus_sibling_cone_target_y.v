module top(a, b, c, clk, y);
  input a, b, c, clk;
  output y;
  wire [1:0] w;
  wire t, u;
  and g1(w[0], a, b);
  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(c), .Q(w[1]));
  xor g2(t, w[0], w[1]);
  not g4(u, t);
  not g5(y, u);
endmodule
