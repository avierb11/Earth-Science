/*
These functions will generally describe the various hydrogeologic
processes I would like to model.
*/

// Broad flow functions that deal with
//------------------------------------------------------------------------------
void GetTransientFlow(float *heads, float *transient_flow, float *multipliers, float timeStep, int dim);
/* Determines the net change in head at each element, storing the value in transient_flow but does NOT apply them*/

void ApplyTransientFlow(float *heads float *transient_flow, int dim);
/* Applies the head differences to the existing heads array and resets the transient_queue */
//------------------------------------------------------------------------------


// Well functions
//------------------------------------------------------------------------------
void WellWithdraw(int x, int y, int z, int well_depth, int widthdraw_amount);
/*Removes the specified water amount from a given location to simulate flow from a  well*/
//------------------------------------------------------------------------------


// Solute transport functions
//------------------------------------------------------------------------------
void Advection(float *heads, float *concentrations, float *transient_solute, float timeDelta);
// Determines advective solute transport

void Dispersion(float *heads, float *concentrations, float *transient_solute, float *dispersivity, float timeDelta);
// Determines solute evolution due only to dispersion

void Diffusion(float *heads, float *concentrations, float *transient_solute, float *diffusivity, float timeDelta)
// Determines solute evolution due only to diffusion
//------------------------------------------------------------------------------
