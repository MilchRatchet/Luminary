#ifndef CU_DIRECTIVES_H
#define CU_DIRECTIVES_H

/*
 * Define WEIGHT_BASED_EXIT for the following to apply
 *
 * Rays with an accumulated weight below CUTOFF are cancelled
 * Higher values provide better performance at the cost of extreme bright lights not being shaded properly
 *
 * Change BRIGHTEST_EMISSION to the intensity of the brightest light for best performance without
 * visual degradation
 *
 * Rays with an accumulated weight below PROBABILISTIC_CUTOFF have a chance equal to
 * the inverse linear interpolation of the weight and CUTOFF and PROBABILISTIC_CUTOFF
 * Higher values provide much better performance at the cost of noisy/too dark areas
 * Setting PROBABILISTIC_CUTOFF >= CUTOFF will deactivate this option
 */
 #define WEIGHT_BASED_EXIT
 #define BRIGHTEST_EMISSION 20.0f
 #define CUTOFF ((1.0f)/(BRIGHTEST_EMISSION * 255.0f))
 #define PROBABILISTIC_CUTOFF ((1.0f)/(255.0f))

 /*
  * Define LOW_QUALITY_LONG_BOUNCES for the following to apply
  *
  * After each bounce there is a 1/(max_depth) chance for the sample to exit early
  * This provides much better performance at the cost of noisy indirectly lit areas
  *
  * MIN_BOUNCES controls the minimum amount of bounces that have to happen before this may happen
  */
 //#define LOW_QUALITY_LONG_BOUNCES
 #define MIN_BOUNCES 1


  /*
  * Define FIRST_LIGHT_ONLY for the following to apply
  *
  * Samples exit early if some light data was already gathered
  */
 #define FIRST_LIGHT_ONLY

#endif /* CU_DIRECTIVES_H */
