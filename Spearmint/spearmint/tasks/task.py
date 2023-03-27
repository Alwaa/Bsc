# -*- coding: utf-8 -*-
# Spearmint
#
# Academic and Non-Commercial Research Use Software License and Terms
# of Use
#
# Spearmint is a software package to perform Bayesian optimization
# according to specific algorithms (the “Software”).  The Software is
# designed to automatically run experiments (thus the code name
# 'spearmint') in a manner that iteratively adjusts a number of
# parameters so as to minimize some objective in as few runs as
# possible.
#
# The Software was developed by Ryan P. Adams, Michael Gelbart, and
# Jasper Snoek at Harvard University, Kevin Swersky at the
# University of Toronto (“Toronto”), and Hugo Larochelle at the
# Université de Sherbrooke (“Sherbrooke”), which assigned its rights
# in the Software to Socpra Sciences et Génie
# S.E.C. (“Socpra”). Pursuant to an inter-institutional agreement
# between the parties, it is distributed for free academic and
# non-commercial research use by the President and Fellows of Harvard
# College (“Harvard”).
#
# Using the Software indicates your agreement to be bound by the terms
# of this Software Use Agreement (“Agreement”). Absent your agreement
# to the terms below, you (the “End User”) have no rights to hold or
# use the Software whatsoever.
#
# Harvard agrees to grant hereunder the limited non-exclusive license
# to End User for the use of the Software in the performance of End
# User’s internal, non-commercial research and academic use at End
# User’s academic or not-for-profit research institution
# (“Institution”) on the following terms and conditions:
#
# 1.  NO REDISTRIBUTION. The Software remains the property Harvard,
# Toronto and Socpra, and except as set forth in Section 4, End User
# shall not publish, distribute, or otherwise transfer or make
# available the Software to any other party.
#
# 2.  NO COMMERCIAL USE. End User shall not use the Software for
# commercial purposes and any such use of the Software is expressly
# prohibited. This includes, but is not limited to, use of the
# Software in fee-for-service arrangements, core facilities or
# laboratories or to provide research services to (or in collaboration
# with) third parties for a fee, and in industry-sponsored
# collaborative research projects where any commercial rights are
# granted to the sponsor. If End User wishes to use the Software for
# commercial purposes or for any other restricted purpose, End User
# must execute a separate license agreement with Harvard.
#
# Requests for use of the Software for commercial purposes, please
# contact:
#
# Office of Technology Development
# Harvard University
# Smith Campus Center, Suite 727E
# 1350 Massachusetts Avenue
# Cambridge, MA 02138 USA
# Telephone: (617) 495-3067
# Facsimile: (617) 495-9568
# E-mail: otd@harvard.edu
#
# 3.  OWNERSHIP AND COPYRIGHT NOTICE. Harvard, Toronto and Socpra own
# all intellectual property in the Software. End User shall gain no
# ownership to the Software. End User shall not remove or delete and
# shall retain in the Software, in any modifications to Software and
# in any Derivative Works, the copyright, trademark, or other notices
# pertaining to Software as provided with the Software.
#
# 4.  DERIVATIVE WORKS. End User may create and use Derivative Works,
# as such term is defined under U.S. copyright laws, provided that any
# such Derivative Works shall be restricted to non-commercial,
# internal research and academic use at End User’s Institution. End
# User may distribute Derivative Works to other Institutions solely
# for the performance of non-commercial, internal research and
# academic use on terms substantially similar to this License and
# Terms of Use.
#
# 5.  FEEDBACK. In order to improve the Software, comments from End
# Users may be useful. End User agrees to provide Harvard with
# feedback on the End User’s use of the Software (e.g., any bugs in
# the Software, the user experience, etc.).  Harvard is permitted to
# use such information provided by End User in making changes and
# improvements to the Software without compensation or an accounting
# to End User.
#
# 6.  NON ASSERT. End User acknowledges that Harvard, Toronto and/or
# Sherbrooke or Socpra may develop modifications to the Software that
# may be based on the feedback provided by End User under Section 5
# above. Harvard, Toronto and Sherbrooke/Socpra shall not be
# restricted in any way by End User regarding their use of such
# information.  End User acknowledges the right of Harvard, Toronto
# and Sherbrooke/Socpra to prepare, publish, display, reproduce,
# transmit and or use modifications to the Software that may be
# substantially similar or functionally equivalent to End User’s
# modifications and/or improvements if any.  In the event that End
# User obtains patent protection for any modification or improvement
# to Software, End User agrees not to allege or enjoin infringement of
# End User’s patent against Harvard, Toronto or Sherbrooke or Socpra,
# or any of the researchers, medical or research staff, officers,
# directors and employees of those institutions.
#
# 7.  PUBLICATION & ATTRIBUTION. End User has the right to publish,
# present, or share results from the use of the Software.  In
# accordance with customary academic practice, End User will
# acknowledge Harvard, Toronto and Sherbrooke/Socpra as the providers
# of the Software and may cite the relevant reference(s) from the
# following list of publications:
#
# Practical Bayesian Optimization of Machine Learning Algorithms
# Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams
# Neural Information Processing Systems, 2012
#
# Multi-Task Bayesian Optimization
# Kevin Swersky, Jasper Snoek and Ryan Prescott Adams
# Advances in Neural Information Processing Systems, 2013
#
# Input Warping for Bayesian Optimization of Non-stationary Functions
# Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams
# Preprint, arXiv:1402.0929, http://arxiv.org/abs/1402.0929, 2013
#
# Bayesian Optimization and Semiparametric Models with Applications to
# Assistive Technology Jasper Snoek, PhD Thesis, University of
# Toronto, 2013
#
# 8.  NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS." TO THE FULLEST
# EXTENT PERMITTED BY LAW, HARVARD, TORONTO AND SHERBROOKE AND SOCPRA
# HEREBY DISCLAIM ALL WARRANTIES OF ANY KIND (EXPRESS, IMPLIED OR
# OTHERWISE) REGARDING THE SOFTWARE, INCLUDING BUT NOT LIMITED TO ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OWNERSHIP, AND NON-INFRINGEMENT.  HARVARD, TORONTO AND
# SHERBROOKE AND SOCPRA MAKE NO WARRANTY ABOUT THE ACCURACY,
# RELIABILITY, COMPLETENESS, TIMELINESS, SUFFICIENCY OR QUALITY OF THE
# SOFTWARE.  HARVARD, TORONTO AND SHERBROOKE AND SOCPRA DO NOT WARRANT
# THAT THE SOFTWARE WILL OPERATE WITHOUT ERROR OR INTERRUPTION.
#
# 9.  LIMITATIONS OF LIABILITY AND REMEDIES. USE OF THE SOFTWARE IS AT
# END USER’S OWN RISK. IF END USER IS DISSATISFIED WITH THE SOFTWARE,
# ITS EXCLUSIVE REMEDY IS TO STOP USING IT.  IN NO EVENT SHALL
# HARVARD, TORONTO OR SHERBROOKE OR SOCPRA BE LIABLE TO END USER OR
# ITS INSTITUTION, IN CONTRACT, TORT OR OTHERWISE, FOR ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR OTHER
# DAMAGES OF ANY KIND WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH
# THE SOFTWARE, EVEN IF HARVARD, TORONTO OR SHERBROOKE OR SOCPRA IS
# NEGLIGENT OR OTHERWISE AT FAULT, AND REGARDLESS OF WHETHER HARVARD,
# TORONTO OR SHERBROOKE OR SOCPRA IS ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
# 10. INDEMNIFICATION. To the extent permitted by law, End User shall
# indemnify, defend and hold harmless Harvard, Toronto and Sherbrooke
# and Socpra, their corporate affiliates, current or future directors,
# trustees, officers, faculty, medical and professional staff,
# employees, students and agents and their respective successors,
# heirs and assigns (the "Indemnitees"), against any liability,
# damage, loss or expense (including reasonable attorney's fees and
# expenses of litigation) incurred by or imposed upon the Indemnitees
# or any one of them in connection with any claims, suits, actions,
# demands or judgments arising from End User’s breach of this
# Agreement or its Institution’s use of the Software except to the
# extent caused by the gross negligence or willful misconduct of
# Harvard, Toronto or Sherbrooke or Socpra. This indemnification
# provision shall survive expiration or termination of this Agreement.
#
# 11. GOVERNING LAW. This Agreement shall be construed and governed by
# the laws of the Commonwealth of Massachusetts regardless of
# otherwise applicable choice of law standards.
#
# 12. NON-USE OF NAME.  Nothing in this License and Terms of Use shall
# be construed as granting End Users or their Institutions any rights
# or licenses to use any trademarks, service marks or logos associated
# with the Software.  You may not use the terms “Harvard” or
# “University of Toronto” or “Université de Sherbrooke” or “Socpra
# Sciences et Génie S.E.C.” (or a substantially similar term) in any
# way that is inconsistent with the permitted uses described
# herein. You agree not to use any name or emblem of Harvard, Toronto
# or Sherbrooke, or any of their subdivisions for any purpose, or to
# falsely suggest any relationship between End User (or its
# Institution) and Harvard, Toronto and/or Sherbrooke, or in any
# manner that would infringe or violate any of their rights.
#
# 13. End User represents and warrants that it has the legal authority
# to enter into this License and Terms of Use on behalf of itself and
# its Institution.


import numpy as np
import sys
import logging
from .input_space import InputSpace

class Task(object):
    """
    A task is a dataset that contains utilities to map
    from the variables specified in a config file to a matrix
    representation that can be used in a chooser/model.
    """
    
    def __init__(self, task_name, task_options, num_dims):
        self.name       = task_name
        self.type       = task_options['type'].lower()
        self.options    = task_options # There may be many options here, like constraint
        # confidence thresholds, etc etc. We do not know exactly what they will be.

        #TODO: Validate the data
        self._inputs  = np.zeros((0, num_dims))
        self._pending = np.zeros((0, num_dims))
        self._values  = np.array([])
        self._costs   = np.array([])
        
        self.durations = np.array([])

        self.standardization_mean     = None
        self.standardization_variance = None

    def has_valid_inputs(self):
        return self.valid_inputs.size > 0
    def has_inputs(self):
        return self._inputs.size > 0
    def has_pending(self):
        return self._pending.size > 0

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def pending(self):
        return self._pending

    @pending.setter
    def pending(self, pending):
        self._pending = pending

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    @property
    def valid_inputs(self):
        return self._inputs[~np.isnan(self._values)]

    @property
    def valid_values(self):
        return self._values[~np.isnan(self._values)]

    def valid_normalized_inputs(self, input_space):
        return input_space.to_unit(self.valid_inputs)

    def normalized_pending(self, input_space):
        if self.pending.shape[0] > 0:
            return input_space.to_unit(self.pending)
        else:
            return None

    def numComplete(self, jobs):
        if not jobs:
            return 0

        return len(filter(lambda job: self.name in job['tasks'] and job['status']=='complete', jobs))

    def numPending(self, jobs):
        if not jobs:
            return 0

        return len(filter(lambda job: self.name in job['tasks'] and job['status']=='pending', jobs))


    def maxCompleteReached(self, jobs):
        return self.numComplete(jobs) >= self.options['max_finished_jobs']

    def valid_normalized_values(self):
        if self.options['likelihood'].lower() in ['binomial', 'step']:
            # If binomial, don't standardize!
            return self.valid_values # COUNTS
        elif self.type == 'objective':
            # If it's a regular objective
            values = self.valid_values
            values = self.standardize_mean(values)
            values = self.standardize_variance(values)
            return values
        elif self.type == 'constraint':
            # definitely don't standardize the mean here. because the constraint value
            # relative to 0 is important
            # scaling is OK, but not by the variance -- that is messed up if you aren't
            # standardizing the mean first (e.g. if your values are 1.11 and 1.12)
            # but dividing by the max of the absolute value is good. it means after scaling,
            # the biggest absolute value should be 1.
            return self.standardize_variance(self.valid_values, use_max=True)


    def standardize_mean(self, y):
        if y.size == 0:
            return y

        mean = y.mean()
        self.standardization_mean = mean
        return y - mean


    def standardize_variance(self, y, use_max=False):
        if y.size == 0:
            return y

        if use_max:
            y_std = np.max(np.abs(y))
        else:
            y_std  = y.std()

        # some weird logic here:
        # first, check if std is 0. if so this indicated that all elements of y are the same
        # (or it just has one element)
        # then, set std to y[0] so y values get normalized to 1
        if y_std == 0:
            y_std = y[0]
        # but what if they were all the same And equal to zero? then don't normalize!
        if y_std == 0:
            y_std = 1.0


        self.standardization_variance = y_std

        return y / y_std

    def unstandardize_mean(self, y):
        if y.size == 0:
            return y

        if self.standardization_mean is None:
            return y
            # return y  # if was never standardized, this means it is a constraint or something
            # raise Exception("values were never standardized")

        return y + self.standardization_mean


    def unstandardize_variance(self, y):  
        if y.size == 0:
            return y # must do this before checking if standardization_variance is None

        if self.standardization_variance is None:
            return y
            # raise Exception("values were never standardized")

        return y * self.standardization_variance


def print_tasks_status(tasks, jobs):
    if len(tasks) == 1:
        # sys.stderr.write('Status: %d pending, %d complete.\n\n'
        #     % (resources[0].numPending(jobs), resources[0].numComplete(jobs)))
        pass
    else:
        left_indent = 16
        
        title = 'Tasks:'
        pad = ' '*(left_indent-len(title))
        sys.stderr.write('\n')
        sys.stderr.write(title)
        sys.stderr.write(pad)

        indentation = ' '*left_indent

        sys.stderr.write('NAME          PENDING    COMPLETE\n')
        sys.stderr.write(indentation)
        sys.stderr.write('----          -------    --------\n')
        totalPending = 0
        totalComplete = 0
        for task in tasks:
            p = task.numPending(jobs)
            c = task.numComplete(jobs)
            totalPending += p
            totalComplete += c
            sys.stderr.write("%s%-12.12s  %-9d  %-10d\n" % (indentation, task.name, p, c))
        sys.stderr.write("%s%-12.12s  %-9d  %-10d\n" % (indentation, '*TOTAL*', totalPending, totalComplete))
        sys.stderr.write('\n')
