{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block summaries %}

   {%- if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}

   {%- if classes %}
   .. rubric:: Classes

   .. autosummary::
      :signatures: short

   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}

   {%- if functions %}
   .. rubric:: Functions

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}

   {%- if attributes %}
   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {% block classes %}

   {%- if classes %}
   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
      :special-members: __call__
      :undoc-members:
      :show-inheritance:
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {% block functions %}

   {%- if functions %}
   {% for item in functions %}
   .. autofunction:: {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {% block exceptions %}
   {%- if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}

   {% for item in exceptions %}
   .. autoexception:: {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}
