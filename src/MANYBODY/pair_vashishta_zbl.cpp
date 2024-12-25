#include "pair_vashishta_zbl.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "potential_file_reader.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static constexpr int DELTA = 4;

/*-----------------------------------------------------*/


PairVashishtaZBL::PairVashishtaZBL(LAMMPS *lmp) : PairVashishta(lmp)
{
  // hard-wired constants in metal or real units
  // a0 = Bohr radius
  // epsilon0 = permittivity of vacuum = q / energy-distance units
  // e = unit charge
  // 1 Kcal/mole = 0.043365121 eV

    global_a_0 = 0.529;
    global_epsilon_0 = 0.00552635;
    global_e = 1.0;
}

/*-----------------------------------------------------*/

void PairVashishtaZBL::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "vashishta/zbl", unit_convert_flag);
    char * line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,
                                                            unit_convert);

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();
        std::string kname = values.next_string();

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement, kelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;
        for (kelement = 0; kelement < nelements; kelement++)
          if (kname == elements[kelement]) break;
        if (kelement == nelements) continue;

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                              "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA*sizeof(Param));
        }

        params[nparams].ielement    = ielement;
        params[nparams].jelement    = jelement;
        params[nparams].kelement    = kelement;
        params[nparams].bigh        = values.next_double();
        params[nparams].eta         = values.next_double();
        params[nparams].zi          = values.next_double();
        params[nparams].zj          = values.next_double();
        params[nparams].lambda1     = values.next_double();
        params[nparams].bigd        = values.next_double();
        params[nparams].lambda4     = values.next_double();
        params[nparams].bigw        = values.next_double();
        params[nparams].cut         = values.next_double();
        params[nparams].bigb        = values.next_double();
        params[nparams].gamma       = values.next_double();
        params[nparams].r0          = values.next_double();
        params[nparams].bigc        = values.next_double();
        params[nparams].costheta    = values.next_double();
        params[nparams].Z_i         = values.next_double();
        params[nparams].Z_j         = values.next_double();
        params[nparams].ZBLcut      = values.next_double();
        params[nparams].ZBLexpscale = values.next_double();

        if (unit_convert) {
          params[nparams].bigh *= conversion_factor;
          params[nparams].bigd *= conversion_factor;
          params[nparams].bigw *= conversion_factor;
          params[nparams].bigb *= conversion_factor;
        }

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      if (params[nparams].bigb < 0.0 || params[nparams].gamma < 0.0 ||
          params[nparams].r0 < 0.0 || params[nparams].bigc < 0.0 ||
          params[nparams].bigh < 0.0 || params[nparams].eta < 0.0 ||
          params[nparams].lambda1 < 0.0 || params[nparams].bigd < 0.0 ||
          params[nparams].lambda4 < 0.0 || params[nparams].bigw < 0.0 ||
          params[nparams].cut < 0.0) ||
          params[nparams].Z_i < 1.0 ||
          params[nparams].Z_j < 1.0 ||
          params[nparams].ZBLcut < 0.0 ||
          params[nparams].ZBLexpscale < 0.0)
        error->one(FLERR,"Illegal Vashishta parameter");

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam*sizeof(Param), MPI_BYTE, 0, world);
}

/*-----------------------------------------------------*/

void PairVashishtaZBL::twobody(Param *param, double rsq, double &fforce,
                            int eflag, double &eng)
{
  double r,rinvsq,r4inv,r6inv,reta,lam1r,lam4r,vc2,vc3;

// Vashishta twobody 
  
  r = sqrt(rsq);
  rinv = 1.0/r;
  rinvsq = 1.0/rsq;
  r4inv = rinvsq*rinvsq;
  r6inv = rinvsq*r4inv;
  reta = pow(r,-param->eta);
  lam1r = r*param->lam1inv;
  lam4r = r*param->lam4inv;
  vc2 = param->zizj * exp(-lam1r)/r;
  vc3 = param->mbigd * r4inv*exp(-lam4r);

  double fforce_att_vash = (param->dvrc*r
            - (4.0*vc3 + lam4r*vc3+param->big6w*r6inv)
            ) * rinvsq;
  double eng_att_vash = - vc3 - param->bigw*r6inv + param->c0 - r*param->dvrc;

  double fforce_rep_vash = - (- param->heta*reta - vc2 - lam1r*vc2)
            ) * rinv;
  double eng_rep_vash = param->bigh*reta + vc2;

  // ZBL repulsive portion

  double esq = square(global_e);
  double a_ij = (0.8854*global_a_0) /
    (pow(param->Z_i,0.23) + pow(param->Z_j,0.23));
  double premult = (param->Z_i * param->Z_j * esq)/(4.0*MY_PI*global_epsilon_0);
  double r_ov_a = r/a_ij;
  double phi = 0.1818*exp(-3.2*r_ov_a) + 0.5099*exp(-0.9423*r_ov_a) +
    0.2802*exp(-0.4029*r_ov_a) + 0.02817*exp(-0.2016*r_ov_a);
  double dphi = (1.0/a_ij) * (-3.2*0.1818*exp(-3.2*r_ov_a) -
                              0.9423*0.5099*exp(-0.9423*r_ov_a) -
                              0.4029*0.2802*exp(-0.4029*r_ov_a) -
                              0.2016*0.02817*exp(-0.2016*r_ov_a));
  double fforce_ZBL = premult*-phi/rsq + premult*dphi/r;
  double eng_ZBL = premult*(1.0/r)*phi;

// Combine two parts with smoothing by Fermi-like function

  fforce = (-F_fermi_d(r,param) * eng_ZBL +
             (1.0 - F_fermi(r,param))*fforce_ZBL +
             F_fermi_d(r,param)*eng_rep_vash + F_fermi(r,param)*fforce_rep_vash) / r +
             fforce_att_vash;

  if (eflag)
    eng = (1.0 - F_fermi(r,param))*eng_ZBL + F_fermi(r,param)*eng_rep_vash + eng_att_vash;

}

/*-----------------------------------------------------*/

/* ----------------------------------------------------------------------
   Fermi-like smoothing function
------------------------------------------------------------------------- */

double PairVashishtaZBL::F_fermi(double r, Param *param)
{
  return 1.0 / (1.0 + exp(-param->ZBLexpscale*(r-param->ZBLcut)));
}

/* ----------------------------------------------------------------------
   Fermi-like smoothing function derivative with respect to r
------------------------------------------------------------------------- */

double PairVashishtaZBL::F_fermi_d(double r, Param *param)
{
  return param->ZBLexpscale*exp(-param->ZBLexpscale*(r-param->ZBLcut)) /
    square(1.0 + exp(-param->ZBLexpscale*(r-param->ZBLcut)));
}
