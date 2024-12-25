#ifdef PAIR_CLASS
// clang-format off
PairStyle(vashishta/zbl,PairVashishtaZBL);
// clang-format on
#else

#ifndef LMP_PAIR_VASHISHTA_ZBL_H
#define LMP_PAIR_VASHISHTA_ZBL_H

#include "pair_vashishta.h"

namespace LAMMPS_NS {

class PairVashishtaZBL : public PairVashishta {
 public:
  PairVashishtaZBL(class LAMMPS *);

  static constexpr int NPARAMS_PER_LINE = 21;

 protected:
  double global_a_0;          // Bohr radius for Coulomb repulsion
  double global_epsilon_0;    // permittivity of vacuum for Coulomb repulsion
  double global_e;            // proton charge (negative of electron charge)

  void read_file(char *) override;
  void twobody(Param *param, double rsq, double &fforce, int eflag, double &eng)

  double F_fermi(double, Param *);
  double F_fermi_d(double, Param *);
};

}    // namespace LAMMPS_NS

#endif
#endif
