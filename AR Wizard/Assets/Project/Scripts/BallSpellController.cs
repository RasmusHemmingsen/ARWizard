using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallSpellController : MonoBehaviour
{
    public GameObject Hand;
    public BallSpell BallSpell;

    private Vector3 mousePos;
    private float offsetDistance;
    private int activeEffect = 0;

    private void Start()
    {
        offsetDistance = Hand.transform.position.z - Camera.main.transform.position.z;
    }

    void Update()
    {
        if (!Hand.activeInHierarchy)
        {
            Debug.Log("Hand activated");
            Hand.SetActive(true);
        }

        MoveHand();
        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            ToggleEffect();
        }
    }

    private void ToggleEffect()
    {
        switch (activeEffect)
        {
            case 0:
                {
                    BallSpell.StartSmoke();
                    activeEffect++;
                    break;
                }
            case 1:
                {
                    BallSpell.StartFire();
                    activeEffect++;
                    break;
                }
            case 2:
                {
                    //BallSpell.Shoot();
                    activeEffect++;
                    break;
                }
            case 3:
                {
                    BallSpell.StartExplosion();
                    activeEffect++;
                    break;
                }
            default:
                {
                    activeEffect = 0;
                    break;
                }
        }
    }

    private void MoveHand()
    {
        mousePos = Input.mousePosition;
        mousePos.z += offsetDistance;
        Hand.transform.position = Camera.main.ScreenToWorldPoint(mousePos);
    }
}